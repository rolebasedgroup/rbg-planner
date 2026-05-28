/*
Copyright 2025 The RBG Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controller

import (
	"context"
	"fmt"
	"strconv"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"

	rbgv1alpha1 "github.com/rolebasedgroup/rbg-planner/api/v1alpha1"
)

const (
	finalizerName = "rolebasedgroup.inference-extension.io/finalizer"

	// Default profiler image hardcoded in the operator.
	defaultProfilerImage = "ghcr.io/rolebasedgroup/rbg-profiler:latest"

	// Default Prometheus endpoint (in-cluster).
	defaultPrometheusEndpoint = "http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090"

	// ClusterRole name shared by all planner instances.
	plannerClusterRoleName = "rbg-planner-role"

	// RBG CRD coordinates for unstructured access.
	rbgGroup   = "workloads.x-k8s.io"
	rbgVersion = "v1alpha2"
	rbgPlural  = "rolebasedgroups"
)

var rbgGVR = schema.GroupVersionResource{Group: rbgGroup, Version: rbgVersion, Resource: rbgPlural}

// RoleAutoScalerReconciler reconciles a RoleAutoScaler object.
type RoleAutoScalerReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Recorder record.EventRecorder
}

// +kubebuilder:rbac:groups=rolebasedgroup.inference-extension.io,resources=roleautoscalers,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=rolebasedgroup.inference-extension.io,resources=roleautoscalers/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=rolebasedgroup.inference-extension.io,resources=roleautoscalers/finalizers,verbs=update
// +kubebuilder:rbac:groups=workloads.x-k8s.io,resources=rolebasedgroups,verbs=get;list;watch
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create;delete
// +kubebuilder:rbac:groups="",resources=serviceaccounts;configmaps,verbs=get;list;watch;create;update;delete
// +kubebuilder:rbac:groups=rbac.authorization.k8s.io,resources=clusterroles;clusterrolebindings,verbs=get;list;watch;create;update;delete

func (r *RoleAutoScalerReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// 1. Fetch the RoleAutoScaler CR.
	ras := &unstructured.Unstructured{}
	ras.SetGroupVersionKind(schema.GroupVersionKind{
		Group:   "rolebasedgroup.inference-extension.io",
		Version: "v1alpha1",
		Kind:    "RoleAutoScaler",
	})
	if err := r.Get(ctx, req.NamespacedName, ras); err != nil {
		if apierrors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	// Parse spec from unstructured — we use typed access via helpers.
	spec, err := parseRASSpec(ras)
	if err != nil {
		logger.Error(err, "failed to parse RoleAutoScaler spec")
		return ctrl.Result{}, r.setPhase(ctx, ras, "Failed", "InvalidSpec", err.Error())
	}

	// 2. Handle deletion — cleanup cluster-scoped resources.
	if ras.GetDeletionTimestamp() != nil {
		if controllerutil.ContainsFinalizer(ras, finalizerName) {
			if err := r.cleanupClusterResources(ctx, ras); err != nil {
				logger.Error(err, "failed to cleanup cluster resources")
				return ctrl.Result{}, err
			}
			controllerutil.RemoveFinalizer(ras, finalizerName)
			if err := r.Update(ctx, ras); err != nil {
				return ctrl.Result{}, err
			}
		}
		return ctrl.Result{}, nil
	}

	// 3. Add finalizer if not present.
	if !controllerutil.ContainsFinalizer(ras, finalizerName) {
		controllerutil.AddFinalizer(ras, finalizerName)
		if err := r.Update(ctx, ras); err != nil {
			return ctrl.Result{}, err
		}
	}

	// 4. Validate: target RBG must exist (same name, same namespace).
	rbg := &unstructured.Unstructured{}
	rbg.SetGroupVersionKind(schema.GroupVersionKind{
		Group:   rbgGroup,
		Version: rbgVersion,
		Kind:    "RoleBasedGroup",
	})
	rbgName := ras.GetName()
	rbgNamespace := ras.GetNamespace()
	if err := r.Get(ctx, types.NamespacedName{Name: rbgName, Namespace: rbgNamespace}, rbg); err != nil {
		if apierrors.IsNotFound(err) {
			logger.Info("target RBG not found", "name", rbgName)
			r.Recorder.Event(ras, corev1.EventTypeWarning, "RBGNotFound",
				fmt.Sprintf("RoleBasedGroup %s/%s not found", rbgNamespace, rbgName))
			_ = r.setPhase(ctx, ras, "Pending", "RBGNotFound",
				fmt.Sprintf("Waiting for RoleBasedGroup %s to exist", rbgName))
			return ctrl.Result{RequeueAfter: 30 * time.Second}, nil
		}
		return ctrl.Result{}, err
	}

	// Read GPU per engine from RBG spec for each role.
	prefillGPUs, decodeGPUs := getGPUsFromRBG(rbg, spec.prefillRoleName, spec.decodeRoleName)

	// 5. Ensure RBAC.
	if err := r.ensureRBAC(ctx, ras); err != nil {
		logger.Error(err, "failed to ensure RBAC")
		return ctrl.Result{}, err
	}

	// 6. Handle Profiling.
	profilingCMName, ready, err := r.handleProfiling(ctx, ras, spec)
	if err != nil {
		logger.Error(err, "profiling failed")
		_ = r.setPhase(ctx, ras, "Failed", "ProfilingFailed", err.Error())
		return ctrl.Result{}, err
	}
	if !ready {
		logger.Info("profiling in progress")
		_ = r.setPhase(ctx, ras, "Initializing", "ProfilingInProgress", "Profiling Job is running")
		return ctrl.Result{RequeueAfter: 30 * time.Second}, nil
	}

	// 7. Ensure Planner Deployment.
	if err := r.ensurePlannerDeployment(ctx, ras, spec, profilingCMName, prefillGPUs, decodeGPUs); err != nil {
		logger.Error(err, "failed to ensure planner deployment")
		return ctrl.Result{}, err
	}

	// 8. Update Status to Ready.
	if err := r.updateReadyStatus(ctx, ras, rbg, spec, profilingCMName); err != nil {
		logger.Error(err, "failed to update status")
		return ctrl.Result{}, err
	}

	logger.Info("reconcile complete", "phase", "Ready")
	return ctrl.Result{RequeueAfter: time.Duration(spec.adjustmentInterval) * time.Second}, nil
}

// parsedSpec holds the parsed RoleAutoScaler spec fields.
type parsedSpec struct {
	adjustmentInterval int
	prefillRoleName    string
	decodeRoleName     string
	prefillMinReplicas int32
	prefillMaxReplicas int32
	decodeMinReplicas  int32
	decodeMaxReplicas  int32
	plannerImage       string
	modelName          string
	ttft               float64
	itl                float64
	loadPredictor      string
	predictionWindow   int
	noCorrection       bool
	dryRun             bool
	profilingImage     string
	metricSource       string
	metricsPort        int
}

func parseRASSpec(ras *unstructured.Unstructured) (*parsedSpec, error) {
	spec, ok := ras.Object["spec"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("missing spec")
	}

	s := &parsedSpec{
		adjustmentInterval: getIntField(spec, "adjustmentInterval", 180),
	}

	// patternOptions.PDDisaggregated
	po, _ := spec["patternOptions"].(map[string]interface{})
	pd, _ := po["PDDisaggregated"].(map[string]interface{})
	if pd == nil {
		return nil, fmt.Errorf("patternOptions.PDDisaggregated is required")
	}
	prefill, _ := pd["prefill"].(map[string]interface{})
	decode, _ := pd["decode"].(map[string]interface{})
	s.prefillRoleName = getStrField(prefill, "roleName", "prefill")
	s.prefillMinReplicas = int32(getIntField(prefill, "minReplicas", 1))
	s.prefillMaxReplicas = int32(getIntField(prefill, "maxReplicas", 1))
	s.decodeRoleName = getStrField(decode, "roleName", "decode")
	s.decodeMinReplicas = int32(getIntField(decode, "minReplicas", 1))
	s.decodeMaxReplicas = int32(getIntField(decode, "maxReplicas", 1))

	// scalerEngine.DynamoPlanner
	se, _ := spec["scalerEngine"].(map[string]interface{})
	dp, _ := se["DynamoPlanner"].(map[string]interface{})
	if dp == nil {
		return nil, fmt.Errorf("scalerEngine.DynamoPlanner is required")
	}
	s.plannerImage = getStrField(dp, "image", "")
	if s.plannerImage == "" {
		return nil, fmt.Errorf("scalerEngine.DynamoPlanner.image is required")
	}
	s.modelName = getStrField(dp, "modelName", "")
	s.ttft = getFloatField(dp, "ttft", 500.0)
	s.itl = getFloatField(dp, "itl", 50.0)
	s.loadPredictor = getStrField(dp, "loadPredictor", "arima")
	s.predictionWindow = getIntField(dp, "predictionWindow", 50)
	s.noCorrection = getBoolField(dp, "noCorrection")
	s.dryRun = getBoolField(dp, "dryRun")

	// profiling
	prof, _ := dp["profiling"].(map[string]interface{})
	s.profilingImage = getStrField(prof, "image", defaultProfilerImage)

	// metricsEndpoint
	me, _ := dp["metricsEndpoint"].(map[string]interface{})
	s.metricSource = getStrField(me, "metricSource", "sglang")
	s.metricsPort = getIntField(me, "port", 9091)

	return s, nil
}

// --- RBAC ---

func (r *RoleAutoScalerReconciler) ensureRBAC(ctx context.Context, ras *unstructured.Unstructured) error {
	name := ras.GetName()
	namespace := ras.GetNamespace()
	saName := name + "-planner"
	crbName := name + "-planner-binding"

	// ServiceAccount
	sa := &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      saName,
			Namespace: namespace,
			OwnerReferences: []metav1.OwnerReference{
				ownerRef(ras),
			},
		},
	}
	if err := r.createOrUpdate(ctx, sa, func() {}); err != nil {
		return fmt.Errorf("ensure ServiceAccount: %w", err)
	}

	// ClusterRole (shared, no owner ref)
	cr := &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name: plannerClusterRoleName,
		},
		Rules: []rbacv1.PolicyRule{
			{
				APIGroups: []string{"workloads.x-k8s.io"},
				Resources: []string{"rolebasedgroups"},
				Verbs:     []string{"get", "list", "patch"},
			},
			{
				APIGroups: []string{"workloads.x-k8s.io"},
				Resources: []string{"rolebasedgroupscalingadapters/scale"},
				Verbs:     []string{"patch"},
			},
		},
	}
	if err := r.createIfNotExists(ctx, cr); err != nil {
		return fmt.Errorf("ensure ClusterRole: %w", err)
	}

	// ClusterRoleBinding (per CR, cleaned via finalizer)
	crb := &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: crbName,
			Labels: map[string]string{
				"app.kubernetes.io/managed-by":            "rbg-planner-operator",
				"rolebasedgroup.inference-extension.io/ras-name":      name,
				"rolebasedgroup.inference-extension.io/ras-namespace": namespace,
			},
		},
		RoleRef: rbacv1.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     plannerClusterRoleName,
		},
		Subjects: []rbacv1.Subject{
			{
				Kind:      "ServiceAccount",
				Name:      saName,
				Namespace: namespace,
			},
		},
	}
	if err := r.createOrUpdate(ctx, crb, func() {
		crb.Subjects = []rbacv1.Subject{
			{Kind: "ServiceAccount", Name: saName, Namespace: namespace},
		}
	}); err != nil {
		return fmt.Errorf("ensure ClusterRoleBinding: %w", err)
	}

	return nil
}

func (r *RoleAutoScalerReconciler) cleanupClusterResources(ctx context.Context, ras *unstructured.Unstructured) error {
	crbName := ras.GetName() + "-planner-binding"
	crb := &rbacv1.ClusterRoleBinding{}
	if err := r.Get(ctx, types.NamespacedName{Name: crbName}, crb); err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}
	return r.Delete(ctx, crb)
}

// --- Profiling ---

func (r *RoleAutoScalerReconciler) handleProfiling(ctx context.Context, ras *unstructured.Unstructured, spec *parsedSpec) (string, bool, error) {
	name := ras.GetName()
	namespace := ras.GetNamespace()
	cmName := name + "-profiling"
	jobName := name + "-profiling"

	// Check if profiling ConfigMap already exists.
	cm := &corev1.ConfigMap{}
	if err := r.Get(ctx, types.NamespacedName{Name: cmName, Namespace: namespace}, cm); err == nil {
		// ConfigMap exists — profiling is complete.
		return cmName, true, nil
	}

	// Check if profiling Job exists.
	job := &batchv1.Job{}
	err := r.Get(ctx, types.NamespacedName{Name: jobName, Namespace: namespace}, job)
	if err != nil {
		if !apierrors.IsNotFound(err) {
			return "", false, err
		}
		// Job does not exist — create it.
		if err := r.createProfilingJob(ctx, ras, spec, jobName, cmName); err != nil {
			return "", false, fmt.Errorf("create profiling job: %w", err)
		}
		return "", false, nil
	}

	// Job exists — check status.
	if job.Status.Succeeded > 0 {
		// Job completed — ConfigMap should have been created by the profiler.
		// Re-check ConfigMap.
		if err := r.Get(ctx, types.NamespacedName{Name: cmName, Namespace: namespace}, cm); err != nil {
			return "", false, fmt.Errorf("profiling job succeeded but ConfigMap %s not found", cmName)
		}
		return cmName, true, nil
	}

	if job.Status.Failed > 0 {
		return "", false, fmt.Errorf("profiling job %s failed", jobName)
	}

	// Still running.
	return "", false, nil
}

func (r *RoleAutoScalerReconciler) createProfilingJob(ctx context.Context, ras *unstructured.Unstructured, spec *parsedSpec, jobName, cmName string) error {
	namespace := ras.GetNamespace()
	rbgName := ras.GetName()
	saName := rbgName + "-planner"

	backoffLimit := int32(1)
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      jobName,
			Namespace: namespace,
			OwnerReferences: []metav1.OwnerReference{
				ownerRef(ras),
			},
		},
		Spec: batchv1.JobSpec{
			BackoffLimit: &backoffLimit,
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					ServiceAccountName: saName,
					RestartPolicy:      corev1.RestartPolicyNever,
					Containers: []corev1.Container{
						{
							Name:  "profiler",
							Image: spec.profilingImage,
							Args: []string{
								"--model-name", spec.modelName,
								"--engine", spec.metricSource,
								"--ttft-sla", fmt.Sprintf("%.1f", spec.ttft),
								"--itl-sla", fmt.Sprintf("%.1f", spec.itl),
								"--rbg-name", rbgName,
								"--namespace", namespace,
								"--output-configmap", cmName,
							},
						},
					},
				},
			},
		},
	}

	return r.Create(ctx, job)
}

// --- Planner Deployment ---

func (r *RoleAutoScalerReconciler) ensurePlannerDeployment(ctx context.Context, ras *unstructured.Unstructured, spec *parsedSpec, profilingCM string, prefillGPUs, decodeGPUs int) error {
	name := ras.GetName()
	namespace := ras.GetNamespace()
	deployName := name + "-planner"
	saName := name + "-planner"

	maxGPUBudget := int(spec.prefillMaxReplicas)*prefillGPUs + int(spec.decodeMaxReplicas)*decodeGPUs

	replicas := int32(1)
	desired := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      deployName,
			Namespace: namespace,
			OwnerReferences: []metav1.OwnerReference{
				ownerRef(ras),
			},
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"app.kubernetes.io/name":     "rbg-planner",
					"app.kubernetes.io/instance": name,
				},
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"app.kubernetes.io/name":     "rbg-planner",
						"app.kubernetes.io/instance": name,
					},
				},
				Spec: corev1.PodSpec{
					ServiceAccountName: saName,
					Containers: []corev1.Container{
						{
							Name:  "planner",
							Image: spec.plannerImage,
							Env:   buildPlannerEnv(name, spec, namespace, maxGPUBudget, prefillGPUs, decodeGPUs),
							VolumeMounts: []corev1.VolumeMount{
								{
									Name:      "profiling",
									MountPath: "/etc/rbg-planner/profiling",
									ReadOnly:  true,
								},
							},
						},
					},
					Volumes: []corev1.Volume{
						{
							Name: "profiling",
							VolumeSource: corev1.VolumeSource{
								ConfigMap: &corev1.ConfigMapVolumeSource{
									LocalObjectReference: corev1.LocalObjectReference{
										Name: profilingCM,
									},
								},
							},
						},
					},
				},
			},
		},
	}

	// Check if Deployment already exists.
	existing := &appsv1.Deployment{}
	err := r.Get(ctx, types.NamespacedName{Name: deployName, Namespace: namespace}, existing)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return r.Create(ctx, desired)
		}
		return err
	}

	// Update if spec changed.
	if !equality.Semantic.DeepEqual(existing.Spec.Template.Spec, desired.Spec.Template.Spec) {
		existing.Spec = desired.Spec
		return r.Update(ctx, existing)
	}

	return nil
}

func buildPlannerEnv(rbgName string, spec *parsedSpec, namespace string, maxGPUBudget, prefillGPUs, decodeGPUs int) []corev1.EnvVar {
	return []corev1.EnvVar{
		{Name: "RBG_NAME", Value: rbgName},
		{Name: "RBG_NAMESPACE", Value: namespace},
		{Name: "PREFILL_ROLE_NAME", Value: spec.prefillRoleName},
		{Name: "DECODE_ROLE_NAME", Value: spec.decodeRoleName},
		{Name: "PROMETHEUS_ENDPOINT", Value: defaultPrometheusEndpoint},
		{Name: "METRIC_SOURCE", Value: spec.metricSource},
		{Name: "MODEL_NAME", Value: spec.modelName},
		{Name: "ADJUSTMENT_INTERVAL", Value: strconv.Itoa(spec.adjustmentInterval)},
		{Name: "MAX_GPU_BUDGET", Value: strconv.Itoa(maxGPUBudget)},
		{Name: "MIN_REPLICAS", Value: strconv.Itoa(int(spec.prefillMinReplicas))},
		{Name: "PREFILL_ENGINE_NUM_GPU", Value: strconv.Itoa(prefillGPUs)},
		{Name: "DECODE_ENGINE_NUM_GPU", Value: strconv.Itoa(decodeGPUs)},
		{Name: "TTFT_SLA", Value: fmt.Sprintf("%.1f", spec.ttft)},
		{Name: "ITL_SLA", Value: fmt.Sprintf("%.1f", spec.itl)},
		{Name: "LOAD_PREDICTOR", Value: spec.loadPredictor},
		{Name: "LOAD_PREDICTION_WINDOW_SIZE", Value: strconv.Itoa(spec.predictionWindow)},
		{Name: "NO_CORRECTION", Value: strconv.FormatBool(spec.noCorrection)},
		{Name: "NO_OPERATION", Value: strconv.FormatBool(spec.dryRun)},
		{Name: "PROFILE_RESULTS_DIR", Value: "/etc/rbg-planner/profiling"},
		{Name: "PLANNER_PROMETHEUS_PORT", Value: strconv.Itoa(spec.metricsPort)},
	}
}

// --- Status ---

func (r *RoleAutoScalerReconciler) updateReadyStatus(ctx context.Context, ras *unstructured.Unstructured, rbg *unstructured.Unstructured, spec *parsedSpec, profilingCM string) error {
	deployName := ras.GetName() + "-planner"

	// Read replica counts from RBG status.
	prefillReplicas := getRoleReadyReplicas(rbg, spec.prefillRoleName)
	decodeReplicas := getRoleReadyReplicas(rbg, spec.decodeRoleName)

	status := map[string]interface{}{
		"phase":              string(PhaseReady),
		"prefillReplicas":    prefillReplicas,
		"decodeReplicas":     decodeReplicas,
		"profilingConfigMap": profilingCM,
		"plannerDeployment":  deployName,
		"lastTransitionTime": metav1.Now().Format(time.RFC3339),
	}

	// Set condition.
	conditions := []interface{}{
		map[string]interface{}{
			"type":               "Ready",
			"status":             "True",
			"lastTransitionTime": metav1.Now().Format(time.RFC3339),
			"reason":             "PlannerRunning",
			"message":            "Planner deployment is running",
		},
	}
	status["conditions"] = conditions

	ras.Object["status"] = status
	return r.Status().Update(ctx, ras)
}

func (r *RoleAutoScalerReconciler) setPhase(ctx context.Context, ras *unstructured.Unstructured, phase, reason, message string) error {
	status, _ := ras.Object["status"].(map[string]interface{})
	if status == nil {
		status = map[string]interface{}{}
	}
	status["phase"] = phase

	conditions := []interface{}{
		map[string]interface{}{
			"type":               "Ready",
			"status":             boolToConditionStatus(phase == string(PhaseReady)),
			"lastTransitionTime": metav1.Now().Format(time.RFC3339),
			"reason":             reason,
			"message":            message,
		},
	}
	status["conditions"] = conditions
	ras.Object["status"] = status

	return r.Status().Update(ctx, ras)
}

// --- Helpers ---

func getGPUsFromRBG(rbg *unstructured.Unstructured, prefillRole, decodeRole string) (int, int) {
	prefillGPUs := 1
	decodeGPUs := 1

	roles, found, _ := unstructured.NestedSlice(rbg.Object, "spec", "roles")
	if !found {
		return prefillGPUs, decodeGPUs
	}

	for _, r := range roles {
		role, ok := r.(map[string]interface{})
		if !ok {
			continue
		}
		roleName, _, _ := unstructured.NestedString(role, "name")
		gpus := extractGPURequest(role)
		if gpus > 0 {
			switch roleName {
			case prefillRole:
				prefillGPUs = gpus
			case decodeRole:
				decodeGPUs = gpus
			}
		}
	}

	return prefillGPUs, decodeGPUs
}

func extractGPURequest(role map[string]interface{}) int {
	// Try spec.roles[].template.spec.containers[].resources.requests["nvidia.com/gpu"]
	containers, found, _ := unstructured.NestedSlice(role, "template", "spec", "containers")
	if !found {
		return 0
	}
	for _, c := range containers {
		container, ok := c.(map[string]interface{})
		if !ok {
			continue
		}
		gpuStr, found, _ := unstructured.NestedString(container, "resources", "requests", "nvidia.com/gpu")
		if found {
			if v, err := strconv.Atoi(gpuStr); err == nil {
				return v
			}
		}
		// Also try as int64 (some k8s serializations).
		gpuInt, found, _ := unstructured.NestedInt64(container, "resources", "requests", "nvidia.com/gpu")
		if found {
			return int(gpuInt)
		}
	}
	return 0
}

func getRoleReadyReplicas(rbg *unstructured.Unstructured, roleName string) int32 {
	statuses, found, _ := unstructured.NestedSlice(rbg.Object, "status", "roleStatuses")
	if !found {
		return 0
	}
	for _, s := range statuses {
		status, ok := s.(map[string]interface{})
		if !ok {
			continue
		}
		name, _, _ := unstructured.NestedString(status, "name")
		if name == roleName {
			ready, _, _ := unstructured.NestedInt64(status, "readyReplicas")
			return int32(ready)
		}
	}
	return 0
}

func ownerRef(ras *unstructured.Unstructured) metav1.OwnerReference {
	blockOwnerDeletion := true
	isController := true
	return metav1.OwnerReference{
		APIVersion:         ras.GetAPIVersion(),
		Kind:               ras.GetKind(),
		Name:               ras.GetName(),
		UID:                ras.GetUID(),
		BlockOwnerDeletion: &blockOwnerDeletion,
		Controller:         &isController,
	}
}

func (r *RoleAutoScalerReconciler) createIfNotExists(ctx context.Context, obj client.Object) error {
	err := r.Create(ctx, obj)
	if apierrors.IsAlreadyExists(err) {
		return nil
	}
	return err
}

func (r *RoleAutoScalerReconciler) createOrUpdate(ctx context.Context, obj client.Object, mutateFn func()) error {
	key := types.NamespacedName{Name: obj.GetName(), Namespace: obj.GetNamespace()}
	existing := obj.DeepCopyObject().(client.Object)
	err := r.Get(ctx, key, existing)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return r.Create(ctx, obj)
		}
		return err
	}
	mutateFn()
	return nil
}

func boolToConditionStatus(b bool) string {
	if b {
		return "True"
	}
	return "False"
}

// Field extraction helpers for unstructured maps.

func getStrField(m map[string]interface{}, key, defaultVal string) string {
	if m == nil {
		return defaultVal
	}
	v, ok := m[key].(string)
	if !ok {
		return defaultVal
	}
	return v
}

func getIntField(m map[string]interface{}, key string, defaultVal int) int {
	if m == nil {
		return defaultVal
	}
	switch v := m[key].(type) {
	case int64:
		return int(v)
	case float64:
		return int(v)
	case int:
		return v
	}
	return defaultVal
}

func getFloatField(m map[string]interface{}, key string, defaultVal float64) float64 {
	if m == nil {
		return defaultVal
	}
	switch v := m[key].(type) {
	case float64:
		return v
	case int64:
		return float64(v)
	}
	return defaultVal
}

func getBoolField(m map[string]interface{}, key string) bool {
	if m == nil {
		return false
	}
	v, _ := m[key].(bool)
	return v
}

// Phase constants for unstructured usage.
const (
	PhaseReady = "Ready"
)

// SetupWithManager sets up the controller with the Manager.
func (r *RoleAutoScalerReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&rbgv1alpha1.RoleAutoScaler{}).
		Owns(&appsv1.Deployment{}).
		Owns(&batchv1.Job{}).
		Named("roleautoscaler").
		Complete(r)
}
