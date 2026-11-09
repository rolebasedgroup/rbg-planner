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
	"path/filepath"
	"strconv"
	"strings"
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
	finalizerName = "inference-extension.rolebasedgroup.io/finalizer"

	// Default planner image hardcoded in the operator.
	defaultPlannerImage = "ghcr.io/rolebasedgroup/rbg-planner:latest"

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

// AutoScalerReconciler reconciles an AutoScaler object.
type AutoScalerReconciler struct {
	client.Client
	Scheme             *runtime.Scheme
	Recorder           record.EventRecorder
	PlannerImage       string
	ProfilerImage      string
	PrometheusEndpoint string
}

// +kubebuilder:rbac:groups=inference-extension.rolebasedgroup.io,resources=autoscalers,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=inference-extension.rolebasedgroup.io,resources=autoscalers/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=inference-extension.rolebasedgroup.io,resources=autoscalers/finalizers,verbs=update
// +kubebuilder:rbac:groups=workloads.x-k8s.io,resources=rolebasedgroups,verbs=get;list;watch;patch
// +kubebuilder:rbac:groups=workloads.x-k8s.io,resources=rolebasedgroupscalingadapters/scale,verbs=patch
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create;delete
// +kubebuilder:rbac:groups="",resources=serviceaccounts;configmaps,verbs=get;list;watch;create;update;delete
// +kubebuilder:rbac:groups="",resources=events,verbs=create;patch
// +kubebuilder:rbac:groups=rbac.authorization.k8s.io,resources=clusterroles;clusterrolebindings,verbs=get;list;watch;create;update;delete

func (r *AutoScalerReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// 1. Fetch the AutoScaler CR.
	ras := &unstructured.Unstructured{}
	ras.SetGroupVersionKind(schema.GroupVersionKind{
		Group:   "inference-extension.rolebasedgroup.io",
		Version: "v1alpha1",
		Kind:    "AutoScaler",
	})
	if err := r.Get(ctx, req.NamespacedName, ras); err != nil {
		if apierrors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	// Parse spec from unstructured — we use typed access via helpers.
	spec, err := r.parseRASSpec(ras)
	if err != nil {
		logger.Error(err, "failed to parse AutoScaler spec")
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
			if err := r.setPhase(ctx, ras, "Pending", "RBGNotFound",
				fmt.Sprintf("Waiting for RoleBasedGroup %s to exist", rbgName)); err != nil {
				logger.Error(err, "failed to set phase")
			}
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

	// 5b. Wait for RBG roles to be ready before profiling.
	prefillReady := getRoleReadyReplicas(rbg, spec.prefillRoleName)
	decodeReady := getRoleReadyReplicas(rbg, spec.decodeRoleName)
	if prefillReady == 0 || decodeReady == 0 {
		logger.Info("RBG roles not ready, waiting",
			"prefillReady", prefillReady, "decodeReady", decodeReady)
		if err := r.setPhase(ctx, ras, "Pending", "RBGNotReady",
			"Waiting for RBG roles to become ready"); err != nil {
			logger.Error(err, "failed to set phase")
		}
		return ctrl.Result{RequeueAfter: 30 * time.Second}, nil
	}

	// 6. Handle Profiling.
	// If profiling previously failed, stop reconciling to avoid loops.
	// The user must delete and recreate the AutoScaler to retry.
	if phase, _, _ := unstructured.NestedString(ras.Object, "status", "phase"); phase == "Failed" {
		return ctrl.Result{}, nil
	}

	profilingCMName, ready, err := r.handleProfiling(ctx, ras, spec)
	if err != nil {
		logger.Error(err, "profiling failed")
		if err := r.setPhase(ctx, ras, "Failed", "ProfilingFailed", err.Error()); err != nil {
			logger.Error(err, "failed to set phase")
		}
		return ctrl.Result{}, nil
	}
	if !ready {
		logger.Info("profiling in progress")
		if err := r.setPhase(ctx, ras, "Initializing", "ProfilingInProgress", "Profiling Job is running"); err != nil {
			logger.Error(err, "failed to set phase")
		}
		return ctrl.Result{RequeueAfter: 5 * time.Minute}, nil
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
	return ctrl.Result{RequeueAfter: time.Duration(spec.scalingInterval) * time.Second}, nil
}

// parsedSpec holds the parsed AutoScaler spec fields.
type parsedSpec struct {
	scalingInterval     int
	backend             string
	prefillRoleName     string
	decodeRoleName      string
	prefillMinReplicas  int32
	prefillMaxReplicas  int32
	decodeMinReplicas   int32
	decodeMaxReplicas   int32
	plannerImage        string
	modelName           string
	ttft                float64
	itl                 float64
	loadPredictor       string
	predictionWindow    int
	noCorrection        bool
	dryRun              bool
	profilingImage      string
	metricSource        string
	metricsPort         int
	prometheusEndpoint  string
	modelCachePVCName   string
	modelCacheModelPath string
	modelCacheMountPath string
	imagePullSecrets    []corev1.LocalObjectReference
}

func (r *AutoScalerReconciler) parseRASSpec(ras *unstructured.Unstructured) (*parsedSpec, error) {
	spec, ok := ras.Object["spec"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("missing spec")
	}

	s := &parsedSpec{
		scalingInterval: getIntField(spec, "scalingInterval", 180),
	}
	s.backend = getStrField(spec, "backend", "sglang-oai")

	// pattern.pdDisaggregated
	po, _ := spec["pattern"].(map[string]interface{})
	pd, _ := po["pdDisaggregated"].(map[string]interface{})
	if pd == nil {
		return nil, fmt.Errorf("pattern.pdDisaggregated is required")
	}
	prefill, _ := pd["prefill"].(map[string]interface{})
	decode, _ := pd["decode"].(map[string]interface{})
	s.prefillRoleName = getStrField(prefill, "roleName", "prefill")
	s.prefillMinReplicas = int32(getIntField(prefill, "minReplicas", 1))
	s.prefillMaxReplicas = int32(getIntField(prefill, "maxReplicas", 1))
	s.decodeRoleName = getStrField(decode, "roleName", "decode")
	s.decodeMinReplicas = int32(getIntField(decode, "minReplicas", 1))
	s.decodeMaxReplicas = int32(getIntField(decode, "maxReplicas", 1))

	// implementation.dynamoPlanner
	se, _ := spec["implementation"].(map[string]interface{})
	dp, _ := se["dynamoPlanner"].(map[string]interface{})
	if dp == nil {
		return nil, fmt.Errorf("implementation.dynamoPlanner is required")
	}
	s.plannerImage = r.PlannerImage
	if s.plannerImage == "" {
		s.plannerImage = defaultPlannerImage
	}
	s.modelName = getStrField(dp, "modelName", "")
	// sla
	slaMap, _ := dp["sla"].(map[string]interface{})
	s.ttft = getFloatField(slaMap, "ttft", 500.0)
	s.itl = getFloatField(slaMap, "itl", 50.0)
	s.loadPredictor = getStrField(dp, "loadPredictor", "arima")
	s.predictionWindow = getIntField(dp, "predictionWindow", 50)
	s.noCorrection = getBoolField(dp, "noCorrection")
	s.dryRun = getBoolField(dp, "dryRun")

	// imagePullSecrets
	if ips, ok := dp["imagePullSecrets"].([]interface{}); ok {
		for _, item := range ips {
			if m, ok := item.(map[string]interface{}); ok {
				if name, ok := m["name"].(string); ok {
					s.imagePullSecrets = append(s.imagePullSecrets, corev1.LocalObjectReference{Name: name})
				}
			}
		}
	}

	// profiling
	prof, _ := dp["profiling"].(map[string]interface{})
	profilerDefault := r.ProfilerImage
	if profilerDefault == "" {
		profilerDefault = defaultProfilerImage
	}
	s.profilingImage = getStrField(prof, "image", profilerDefault)

	// modelSource
	if mc, ok := prof["modelSource"].(map[string]interface{}); ok && mc != nil {
		s.modelCachePVCName = getStrField(mc, "pvcName", "")
		s.modelCacheModelPath = getStrField(mc, "modelPath", "")
		s.modelCacheMountPath = getStrField(mc, "mountPath", "/opt/model")
	}

	// metrics
	me, _ := dp["metrics"].(map[string]interface{})
	s.metricSource = getStrField(me, "metricSource", "")
	s.metricsPort = getIntField(me, "port", 9091)

	// prometheusEndpoint
	s.prometheusEndpoint = r.PrometheusEndpoint
	if s.prometheusEndpoint == "" {
		s.prometheusEndpoint = defaultPrometheusEndpoint
	}

	return s, nil
}

// --- RBAC ---

func (r *AutoScalerReconciler) ensureRBAC(ctx context.Context, ras *unstructured.Unstructured) error {
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
			{
				APIGroups: []string{""},
				Resources: []string{"configmaps"},
				Verbs:     []string{"get", "list", "create", "update"},
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
				"app.kubernetes.io/managed-by":                       "rbg-planner-operator",
				"inference-extension.rolebasedgroup.io/as-name":      name,
				"inference-extension.rolebasedgroup.io/as-namespace": namespace,
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

func (r *AutoScalerReconciler) cleanupClusterResources(ctx context.Context, ras *unstructured.Unstructured) error {
	namespace := ras.GetNamespace()
	name := ras.GetName()

	// 1. Delete profiling Job with foreground cascade so its pods are
	// removed before the Job object disappears from etcd.
	jobName := name + "-profiling"
	job := &batchv1.Job{}
	if err := r.Get(ctx, types.NamespacedName{Name: jobName, Namespace: namespace}, job); err != nil {
		if !apierrors.IsNotFound(err) {
			return fmt.Errorf("get profiling job %s: %w", jobName, err)
		}
		// Job already gone — fall through to CRB cleanup.
	} else {
		if job.GetDeletionTimestamp() == nil {
			if delErr := r.Delete(ctx, job, client.PropagationPolicy(metav1.DeletePropagationForeground)); delErr != nil && !apierrors.IsNotFound(delErr) {
				return fmt.Errorf("delete profiling job %s: %w", jobName, delErr)
			}
		}
		// Job still exists (being deleted) — requeue until it is gone.
		return fmt.Errorf("waiting for profiling job %s deletion", jobName)
	}

	// 2. Delete ClusterRoleBinding.
	crbName := name + "-planner-binding"
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

func (r *AutoScalerReconciler) handleProfiling(ctx context.Context, ras *unstructured.Unstructured, spec *parsedSpec) (string, bool, error) {
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
		if job.GetDeletionTimestamp() == nil {
			if delErr := r.Delete(ctx, job, client.PropagationPolicy(metav1.DeletePropagationForeground)); delErr != nil && !apierrors.IsNotFound(delErr) {
				return "", false, fmt.Errorf("profiling job %s failed and cleanup failed: %w", jobName, delErr)
			}
		}
		return "", false, fmt.Errorf("profiling job %s failed", jobName)
	}

	// Still running.
	return "", false, nil
}

// buildProfilingVolumes constructs PVC volume, volume mount, and tokenizer
// path from the ModelSource spec fields. Returns empty values when pvcName
// is empty.
func buildProfilingVolumes(pvcName, modelPath, mountPath string) (volumes []corev1.Volume, volumeMounts []corev1.VolumeMount, tokenizerPath string) {
	if pvcName == "" {
		return nil, nil, ""
	}
	if mountPath == "" {
		mountPath = "/opt/model"
	}
	volumes = []corev1.Volume{
		{
			Name: "model-cache",
			VolumeSource: corev1.VolumeSource{
				PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
					ClaimName: pvcName,
				},
			},
		},
	}
	volumeMounts = []corev1.VolumeMount{
		{
			Name:      "model-cache",
			MountPath: mountPath,
			ReadOnly:  true,
		},
	}
	if modelPath != "" {
		if strings.HasPrefix(modelPath, "/") {
			tokenizerPath = modelPath
		} else {
			tokenizerPath = filepath.Join(mountPath, modelPath)
		}
	}
	return volumes, volumeMounts, tokenizerPath
}

func (r *AutoScalerReconciler) createProfilingJob(ctx context.Context, ras *unstructured.Unstructured, spec *parsedSpec, jobName, cmName string) error {
	namespace := ras.GetNamespace()
	rbgName := ras.GetName()
	saName := rbgName + "-planner"

	volumes, volumeMounts, tokenizerPath := buildProfilingVolumes(
		spec.modelCachePVCName,
		spec.modelCacheModelPath,
		spec.modelCacheMountPath,
	)

	args := []string{
		"--rbg-name", rbgName,
		"--namespace", namespace,
		"--model-name", spec.modelName,
		"--output-configmap", cmName,
		"--prefill-role-name", spec.prefillRoleName,
		"--decode-role-name", spec.decodeRoleName,
		"--backend", spec.backend,
	}
	if tokenizerPath != "" {
		args = append(args, "--tokenizer", tokenizerPath)
	}

	backoffLimit := int32(1)
	activeDeadline := int64(1800)
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      jobName,
			Namespace: namespace,
			OwnerReferences: []metav1.OwnerReference{
				ownerRef(ras),
			},
		},
		Spec: batchv1.JobSpec{
			BackoffLimit:          &backoffLimit,
			ActiveDeadlineSeconds: &activeDeadline,
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					ServiceAccountName: saName,
					RestartPolicy:      corev1.RestartPolicyNever,
					ImagePullSecrets:   spec.imagePullSecrets,
					Volumes:            volumes,
					Containers: []corev1.Container{
						{
							Name:         "profiler",
							Image:        spec.profilingImage,
							Args:         args,
							VolumeMounts: volumeMounts,
						},
					},
				},
			},
		},
	}

	return r.Create(ctx, job)
}

// --- Planner Deployment ---

func (r *AutoScalerReconciler) ensurePlannerDeployment(ctx context.Context, ras *unstructured.Unstructured, spec *parsedSpec, profilingCM string, prefillGPUs, decodeGPUs int) error {
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
					ImagePullSecrets:   spec.imagePullSecrets,
					Containers: []corev1.Container{
						{
							Name:  "planner",
							Image: spec.plannerImage,
							Env:   buildPlannerEnv(name, spec, namespace, maxGPUBudget, prefillGPUs, decodeGPUs),
							Ports: []corev1.ContainerPort{
								{Name: "metrics", ContainerPort: int32(spec.metricsPort), Protocol: corev1.ProtocolTCP},
							},
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
	env := []corev1.EnvVar{
		{Name: "RBG_NAME", Value: rbgName},
		{Name: "RBG_NAMESPACE", Value: namespace},
		{Name: "PREFILL_ROLE_NAME", Value: spec.prefillRoleName},
		{Name: "DECODE_ROLE_NAME", Value: spec.decodeRoleName},
		{Name: "PROMETHEUS_ENDPOINT", Value: spec.prometheusEndpoint},
		{Name: "BACKEND", Value: spec.backend},
		{Name: "MODEL_NAME", Value: spec.modelName},
		{Name: "ADJUSTMENT_INTERVAL", Value: strconv.Itoa(spec.scalingInterval)},
		{Name: "MAX_GPU_BUDGET", Value: strconv.Itoa(maxGPUBudget)},
		{Name: "MIN_REPLICAS", Value: strconv.Itoa(int(spec.prefillMinReplicas))},
		{Name: "MAX_PREFILL_REPLICAS", Value: strconv.Itoa(int(spec.prefillMaxReplicas))},
		{Name: "MAX_DECODE_REPLICAS", Value: strconv.Itoa(int(spec.decodeMaxReplicas))},
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
	// Only set METRIC_SOURCE when explicitly configured; otherwise the
	// planner derives it from BACKEND (e.g. "sglang-oai" -> "sglang").
	if spec.metricSource != "" {
		env = append(env, corev1.EnvVar{Name: "METRIC_SOURCE", Value: spec.metricSource})
	}
	return env
}

// --- Status ---

func (r *AutoScalerReconciler) updateReadyStatus(ctx context.Context, ras *unstructured.Unstructured, rbg *unstructured.Unstructured, spec *parsedSpec, profilingCM string) error {
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

func (r *AutoScalerReconciler) setPhase(ctx context.Context, ras *unstructured.Unstructured, phase, reason, message string) error {
	if existing, _ := ras.Object["status"].(map[string]interface{}); existing != nil {
		if existingPhase, _ := existing["phase"].(string); existingPhase == phase {
			if conds, _ := existing["conditions"].([]interface{}); len(conds) > 0 {
				for _, c := range conds {
					cond, _ := c.(map[string]interface{})
					if cond == nil || cond["type"] != "Ready" {
						continue
					}
					if cond["reason"] == reason && cond["message"] == message {
						return nil
					}
				}
			}
		}
	}

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

// getRolePodSpec extracts the pod spec from an RBG role, trying multiple
// pattern paths: standalonePattern.template, leaderWorkerPattern.leaderTemplate,
// customComponentsPattern.components[0].template, and a direct template fallback.
// For customComponents, it prefers a component named "engine" or "server"
// (matching the Python discoverer's _get_template logic), then falls back to
// the first component.
func getRolePodSpec(role map[string]interface{}) map[string]interface{} {
	if spec, found, _ := unstructured.NestedMap(role, "standalonePattern", "template", "spec"); found {
		return spec
	}
	if spec, found, _ := unstructured.NestedMap(role, "leaderWorkerPattern", "leaderTemplate", "spec"); found {
		return spec
	}
	components, found, _ := unstructured.NestedSlice(role, "customComponentsPattern", "components")
	if found && len(components) > 0 {
		// Prefer "engine" or "server" component, matching Python _get_template.
		for _, c := range components {
			comp, ok := c.(map[string]interface{})
			if !ok {
				continue
			}
			name, _, _ := unstructured.NestedString(comp, "name")
			if name == "engine" || name == "server" {
				if spec, found, _ := unstructured.NestedMap(comp, "template", "spec"); found {
					return spec
				}
			}
		}
		// Fall back to first component.
		if comp, ok := components[0].(map[string]interface{}); ok {
			if spec, found, _ := unstructured.NestedMap(comp, "template", "spec"); found {
				return spec
			}
		}
	}
	if spec, found, _ := unstructured.NestedMap(role, "template", "spec"); found {
		return spec
	}
	return nil
}

func extractGPURequest(role map[string]interface{}) int {
	podSpec := getRolePodSpec(role)
	if podSpec == nil {
		return 0
	}
	containers, found, _ := unstructured.NestedSlice(podSpec, "containers")
	if !found {
		return 0
	}
	for _, c := range containers {
		container, ok := c.(map[string]interface{})
		if !ok {
			continue
		}
		requests, found, _ := unstructured.NestedMap(container, "resources", "requests")
		if !found {
			continue
		}
		for key, value := range requests {
			if !strings.HasSuffix(key, "/gpu") {
				continue
			}
			switch v := value.(type) {
			case string:
				if n, err := strconv.Atoi(v); err == nil {
					return n
				}
			case int64:
				return int(v)
			case float64:
				return int(v)
			}
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

func (r *AutoScalerReconciler) createIfNotExists(ctx context.Context, obj client.Object) error {
	err := r.Create(ctx, obj)
	if apierrors.IsAlreadyExists(err) {
		return nil
	}
	return err
}

func (r *AutoScalerReconciler) createOrUpdate(ctx context.Context, obj client.Object, mutateFn func()) error {
	key := types.NamespacedName{Name: obj.GetName(), Namespace: obj.GetNamespace()}
	err := r.Get(ctx, key, obj)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return r.Create(ctx, obj)
		}
		return err
	}
	mutateFn()
	return r.Update(ctx, obj)
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
func (r *AutoScalerReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&rbgv1alpha1.AutoScaler{}).
		Owns(&appsv1.Deployment{}).
		Owns(&batchv1.Job{}).
		Named("autoscaler").
		Complete(r)
}
