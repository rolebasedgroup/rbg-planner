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
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/yaml"
)

func TestBuildProfilingVolumes_WithPVCAndRelativeModelPath(t *testing.T) {
	volumes, mounts, tokenizer := buildProfilingVolumes("model-pvc", "deepseek-r1", "/opt/model")

	if len(volumes) != 1 {
		t.Fatalf("expected 1 volume, got %d", len(volumes))
	}
	if volumes[0].PersistentVolumeClaim == nil {
		t.Fatal("expected PersistentVolumeClaim volume source")
	}
	if volumes[0].PersistentVolumeClaim.ClaimName != "model-pvc" {
		t.Errorf("expected claim name 'model-pvc', got %q", volumes[0].PersistentVolumeClaim.ClaimName)
	}

	if len(mounts) != 1 {
		t.Fatalf("expected 1 mount, got %d", len(mounts))
	}
	if mounts[0].MountPath != "/opt/model" {
		t.Errorf("expected mount path '/opt/model', got %q", mounts[0].MountPath)
	}
	if !mounts[0].ReadOnly {
		t.Error("expected ReadOnly mount")
	}

	if tokenizer != "/opt/model/deepseek-r1" {
		t.Errorf("expected tokenizer '/opt/model/deepseek-r1', got %q", tokenizer)
	}
}

func TestBuildProfilingVolumes_WithAbsoluteModelPath(t *testing.T) {
	_, _, tokenizer := buildProfilingVolumes("model-pvc", "/abs/path/model", "/opt/model")

	if tokenizer != "/abs/path/model" {
		t.Errorf("expected tokenizer '/abs/path/model', got %q", tokenizer)
	}
}

func TestBuildProfilingVolumes_WithoutPVCName(t *testing.T) {
	volumes, mounts, tokenizer := buildProfilingVolumes("", "deepseek-r1", "/opt/model")

	if len(volumes) != 0 {
		t.Errorf("expected 0 volumes, got %d", len(volumes))
	}
	if len(mounts) != 0 {
		t.Errorf("expected 0 mounts, got %d", len(mounts))
	}
	if tokenizer != "" {
		t.Errorf("expected empty tokenizer, got %q", tokenizer)
	}
}

func TestBuildProfilingVolumes_WithoutModelPath(t *testing.T) {
	volumes, mounts, tokenizer := buildProfilingVolumes("model-pvc", "", "/opt/model")

	if len(volumes) != 1 {
		t.Fatalf("expected 1 volume (PVC still mounted), got %d", len(volumes))
	}
	if len(mounts) != 1 {
		t.Fatalf("expected 1 mount, got %d", len(mounts))
	}
	if tokenizer != "" {
		t.Errorf("expected empty tokenizer when modelPath is empty, got %q", tokenizer)
	}
}

func TestBuildProfilingVolumes_DefaultMountPath(t *testing.T) {
	_, mounts, _ := buildProfilingVolumes("model-pvc", "model", "")

	if mounts[0].MountPath != "/opt/model" {
		t.Errorf("expected default mount path '/opt/model', got %q", mounts[0].MountPath)
	}
}

func TestExtractGPURequest_StandaloneNvidia(t *testing.T) {
	role := map[string]interface{}{
		"standalonePattern": map[string]interface{}{
			"template": map[string]interface{}{
				"spec": map[string]interface{}{
					"containers": []interface{}{
						map[string]interface{}{
							"resources": map[string]interface{}{
								"requests": map[string]interface{}{
									"nvidia.com/gpu": int64(4),
								},
							},
						},
					},
				},
			},
		},
	}
	if got := extractGPURequest(role); got != 4 {
		t.Errorf("expected 4 GPUs, got %d", got)
	}
}

func TestExtractGPURequest_StandaloneAMD(t *testing.T) {
	role := map[string]interface{}{
		"standalonePattern": map[string]interface{}{
			"template": map[string]interface{}{
				"spec": map[string]interface{}{
					"containers": []interface{}{
						map[string]interface{}{
							"resources": map[string]interface{}{
								"requests": map[string]interface{}{
									"amd.com/gpu": "2",
								},
							},
						},
					},
				},
			},
		},
	}
	if got := extractGPURequest(role); got != 2 {
		t.Errorf("expected 2 GPUs, got %d", got)
	}
}

func TestExtractGPURequest_LeaderWorker(t *testing.T) {
	role := map[string]interface{}{
		"leaderWorkerPattern": map[string]interface{}{
			"leaderTemplate": map[string]interface{}{
				"spec": map[string]interface{}{
					"containers": []interface{}{
						map[string]interface{}{
							"resources": map[string]interface{}{
								"requests": map[string]interface{}{
									"nvidia.com/gpu": float64(8),
								},
							},
						},
					},
				},
			},
		},
	}
	if got := extractGPURequest(role); got != 8 {
		t.Errorf("expected 8 GPUs, got %d", got)
	}
}

func TestExtractGPURequest_CustomComponents(t *testing.T) {
	role := map[string]interface{}{
		"customComponentsPattern": map[string]interface{}{
			"components": []interface{}{
				map[string]interface{}{
					"name": "engine",
					"template": map[string]interface{}{
						"spec": map[string]interface{}{
							"containers": []interface{}{
								map[string]interface{}{
									"resources": map[string]interface{}{
										"requests": map[string]interface{}{
											"nvidia.com/gpu": int64(2),
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}
	if got := extractGPURequest(role); got != 2 {
		t.Errorf("expected 2 GPUs, got %d", got)
	}
}

func TestExtractGPURequest_NoGPU(t *testing.T) {
	role := map[string]interface{}{
		"standalonePattern": map[string]interface{}{
			"template": map[string]interface{}{
				"spec": map[string]interface{}{
					"containers": []interface{}{
						map[string]interface{}{
							"resources": map[string]interface{}{
								"requests": map[string]interface{}{
									"cpu": "4",
								},
							},
						},
					},
				},
			},
		},
	}
	if got := extractGPURequest(role); got != 0 {
		t.Errorf("expected 0 GPUs, got %d", got)
	}
}

func TestExtractGPURequest_NoPattern(t *testing.T) {
	role := map[string]interface{}{}
	if got := extractGPURequest(role); got != 0 {
		t.Errorf("expected 0 GPUs for empty role, got %d", got)
	}
}

func TestGetGPUsFromRBG(t *testing.T) {
	rbg := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"spec": map[string]interface{}{
				"roles": []interface{}{
					map[string]interface{}{
						"name": "prefill",
						"standalonePattern": map[string]interface{}{
							"template": map[string]interface{}{
								"spec": map[string]interface{}{
									"containers": []interface{}{
										map[string]interface{}{
											"resources": map[string]interface{}{
												"requests": map[string]interface{}{
													"nvidia.com/gpu": int64(4),
												},
											},
										},
									},
								},
							},
						},
					},
					map[string]interface{}{
						"name": "decode",
						"standalonePattern": map[string]interface{}{
							"template": map[string]interface{}{
								"spec": map[string]interface{}{
									"containers": []interface{}{
										map[string]interface{}{
											"resources": map[string]interface{}{
												"requests": map[string]interface{}{
													"nvidia.com/gpu": int64(2),
												},
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	prefillGPUs, decodeGPUs := getGPUsFromRBG(rbg, "prefill", "decode")
	if prefillGPUs != 4 {
		t.Errorf("expected prefillGPUs=4, got %d", prefillGPUs)
	}
	if decodeGPUs != 2 {
		t.Errorf("expected decodeGPUs=2, got %d", decodeGPUs)
	}
}

func TestGetGPUsFromRBG_NoRoles(t *testing.T) {
	rbg := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"spec": map[string]interface{}{},
		},
	}

	prefillGPUs, decodeGPUs := getGPUsFromRBG(rbg, "prefill", "decode")
	if prefillGPUs != 1 {
		t.Errorf("expected default prefillGPUs=1, got %d", prefillGPUs)
	}
	if decodeGPUs != 1 {
		t.Errorf("expected default decodeGPUs=1, got %d", decodeGPUs)
	}
}

func TestGetRolePodSpec_CustomComponentsPrefersEngine(t *testing.T) {
	role := map[string]interface{}{
		"customComponentsPattern": map[string]interface{}{
			"components": []interface{}{
				map[string]interface{}{
					"name": "init",
					"template": map[string]interface{}{
						"spec": map[string]interface{}{
							"containers": []interface{}{
								map[string]interface{}{"name": "init-container"},
							},
						},
					},
				},
				map[string]interface{}{
					"name": "engine",
					"template": map[string]interface{}{
						"spec": map[string]interface{}{
							"containers": []interface{}{
								map[string]interface{}{"name": "engine-container"},
							},
						},
					},
				},
			},
		},
	}
	spec := getRolePodSpec(role)
	if spec == nil {
		t.Fatal("expected non-nil pod spec")
	}
	containers, _, _ := unstructured.NestedSlice(spec, "containers")
	if len(containers) != 1 {
		t.Fatalf("expected 1 container from engine component, got %d", len(containers))
	}
	container, _ := containers[0].(map[string]interface{})
	if name, _ := container["name"].(string); name != "engine-container" {
		t.Errorf("expected 'engine-container', got %q", name)
	}
}

// --- handleProfiling deletion tests ---

func TestHandleProfiling_FailedJobTriggersDelete(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := batchv1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add batch scheme: %v", err)
	}
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add core scheme: %v", err)
	}

	ras := &unstructured.Unstructured{}
	ras.SetName("test-rbg")
	ras.SetNamespace("test-ns")

	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-rbg-profiling",
			Namespace: "test-ns",
		},
		Status: batchv1.JobStatus{Failed: 1},
	}

	cl := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(job).
		Build()

	r := &AutoScalerReconciler{Client: cl, Scheme: scheme}

	_, _, err := r.handleProfiling(context.Background(), ras, &parsedSpec{})
	if err == nil {
		t.Fatal("expected error for failed job, got nil")
	}
	if !strings.Contains(err.Error(), "failed") {
		t.Errorf("expected error to contain 'failed', got: %v", err)
	}

	// Job should have been deleted.
	err = cl.Get(context.Background(),
		types.NamespacedName{Name: "test-rbg-profiling", Namespace: "test-ns"},
		&batchv1.Job{})
	if !apierrors.IsNotFound(err) {
		t.Errorf("expected job to be deleted, got error: %v", err)
	}
}

func TestHandleProfiling_FailedJobAlreadyDeleting_ReturnsNotReady(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := batchv1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add batch scheme: %v", err)
	}
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add core scheme: %v", err)
	}

	ras := &unstructured.Unstructured{}
	ras.SetName("test-rbg")
	ras.SetNamespace("test-ns")

	deletionTime := metav1.Now()
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "test-rbg-profiling",
			Namespace:         "test-ns",
			DeletionTimestamp: &deletionTime,
			Finalizers:        []string{"foregroundDeletion"},
		},
		Status: batchv1.JobStatus{Failed: 1},
	}

	cl := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(job).
		Build()

	r := &AutoScalerReconciler{Client: cl, Scheme: scheme}

	_, ready, err := r.handleProfiling(context.Background(), ras, &parsedSpec{})
	if err != nil {
		t.Fatalf("expected no error for deleting job, got: %v", err)
	}
	if ready {
		t.Error("expected not-ready for deleting job")
	}

	// Job should still exist, unchanged — no second delete issued.
	got := &batchv1.Job{}
	if err := cl.Get(context.Background(),
		types.NamespacedName{Name: "test-rbg-profiling", Namespace: "test-ns"}, got); err != nil {
		t.Fatalf("expected job to still exist, got error: %v", err)
	}
	if got.GetDeletionTimestamp() == nil {
		t.Error("expected deletionTimestamp to still be set")
	}
}

// --- cleanupClusterResources tests ---

func TestCleanupClusterResources_DeletesProfilingJobAndWaits(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := batchv1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add batch scheme: %v", err)
	}
	if err := rbacv1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add rbac scheme: %v", err)
	}
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add core scheme: %v", err)
	}

	ras := &unstructured.Unstructured{}
	ras.SetName("test-rbg")
	ras.SetNamespace("test-ns")

	// The fake client doesn't implement propagation policy. A finalizer
	// makes it set deletionTimestamp instead of removing the object,
	// simulating the foreground-cascade behaviour we rely on.
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "test-rbg-profiling",
			Namespace:  "test-ns",
			Finalizers: []string{"foregroundDeletion"},
		},
	}
	// Namespaced RBAC is garbage-collected via ownerRefs; cleanup must not
	// delete it directly.
	roleBinding := &rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-rbg-planner",
			Namespace: "test-ns",
		},
	}

	cl := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(job, roleBinding).
		Build()

	r := &AutoScalerReconciler{Client: cl, Scheme: scheme}

	err := r.cleanupClusterResources(context.Background(), ras)
	if err == nil {
		t.Fatal("expected 'waiting' error, got nil")
	}
	if !strings.Contains(err.Error(), "waiting") {
		t.Errorf("expected error to contain 'waiting', got: %v", err)
	}

	// Job should have deletionTimestamp set (not removed).
	got := &batchv1.Job{}
	if err := cl.Get(context.Background(),
		types.NamespacedName{Name: "test-rbg-profiling", Namespace: "test-ns"}, got); err != nil {
		t.Fatalf("expected job to still exist, got error: %v", err)
	}
	if got.GetDeletionTimestamp() == nil {
		t.Error("expected deletionTimestamp on job")
	}

	// RoleBinding is GC-owned — cleanup must not proceed past the Job.
	if err := cl.Get(context.Background(),
		types.NamespacedName{Name: "test-rbg-planner", Namespace: "test-ns"},
		&rbacv1.RoleBinding{}); err != nil {
		t.Errorf("expected RoleBinding to still exist, got error: %v", err)
	}
}

func TestCleanupClusterResources_JobGoneLeavesNamespacedRBACToGC(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := batchv1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add batch scheme: %v", err)
	}
	if err := rbacv1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add rbac scheme: %v", err)
	}
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add core scheme: %v", err)
	}

	ras := &unstructured.Unstructured{}
	ras.SetName("test-rbg")
	ras.SetNamespace("test-ns")

	roleBinding := &rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-rbg-planner",
			Namespace: "test-ns",
		},
	}

	cl := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(roleBinding).
		Build()

	r := &AutoScalerReconciler{Client: cl, Scheme: scheme}

	if err := r.cleanupClusterResources(context.Background(), ras); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}

	// RoleBinding must survive — namespaced RBAC is reclaimed by GC via
	// ownerRefs, not by the operator.
	if err := cl.Get(context.Background(),
		types.NamespacedName{Name: "test-rbg-planner", Namespace: "test-ns"},
		&rbacv1.RoleBinding{}); err != nil {
		t.Errorf("expected RoleBinding to still exist, got error: %v", err)
	}
}

// --- ensureRBAC tests ---

func findRule(rules []rbacv1.PolicyRule, resource string) *rbacv1.PolicyRule {
	for i := range rules {
		for _, res := range rules[i].Resources {
			if res == resource {
				return &rules[i]
			}
		}
	}
	return nil
}

func hasVerb(verbs []string, verb string) bool {
	for _, v := range verbs {
		if v == verb {
			return true
		}
	}
	return false
}

func newRBACTestScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	scheme := runtime.NewScheme()
	for _, add := range []func(*runtime.Scheme) error{corev1.AddToScheme, rbacv1.AddToScheme} {
		if err := add(scheme); err != nil {
			t.Fatalf("failed to build scheme: %v", err)
		}
	}
	return scheme
}

func newTestRAS() *unstructured.Unstructured {
	ras := &unstructured.Unstructured{}
	ras.SetAPIVersion("inference-extension.rolebasedgroup.io/v1alpha1")
	ras.SetKind("AutoScaler")
	ras.SetName("test-rbg")
	ras.SetNamespace("test-ns")
	ras.SetUID("ras-uid")
	return ras
}

func TestEnsureRBAC_CreatesSeparatePlannerAndProfilerIdentities(t *testing.T) {
	scheme := newRBACTestScheme(t)
	ras := newTestRAS()

	cl := fake.NewClientBuilder().WithScheme(scheme).Build()
	r := &AutoScalerReconciler{Client: cl, Scheme: scheme}

	spec := &parsedSpec{prefillRoleName: "p-engine", decodeRoleName: "d-engine"}
	if err := r.ensureRBAC(context.Background(), ras, spec); err != nil {
		t.Fatalf("ensureRBAC failed: %v", err)
	}

	// Two distinct ServiceAccounts, both owner-referenced to the CR.
	for _, saName := range []string{"test-rbg-planner", "test-rbg-profiler"} {
		sa := &corev1.ServiceAccount{}
		if err := cl.Get(context.Background(),
			types.NamespacedName{Name: saName, Namespace: "test-ns"}, sa); err != nil {
			t.Fatalf("expected ServiceAccount %s to exist, got: %v", saName, err)
		}
		if len(sa.OwnerReferences) != 1 || sa.OwnerReferences[0].UID != "ras-uid" {
			t.Errorf("ServiceAccount %s must be owner-referenced to the AutoScaler, got %+v", saName, sa.OwnerReferences)
		}
	}

	// Planner Role: RBG get/patch on the exact RBG name, scale patch on the
	// two role-specific adapters, and nothing else.
	plannerRole := &rbacv1.Role{}
	if err := cl.Get(context.Background(),
		types.NamespacedName{Name: "test-rbg-planner", Namespace: "test-ns"}, plannerRole); err != nil {
		t.Fatalf("expected planner Role to exist, got: %v", err)
	}
	rbgRule := findRule(plannerRole.Rules, "rolebasedgroups")
	if rbgRule == nil {
		t.Fatalf("planner Role must grant rolebasedgroups access, got %+v", plannerRole.Rules)
	}
	if !hasVerb(rbgRule.Verbs, "get") || !hasVerb(rbgRule.Verbs, "patch") || hasVerb(rbgRule.Verbs, "list") {
		t.Errorf("planner rolebasedgroups verbs must be exactly get;patch, got %v", rbgRule.Verbs)
	}
	if len(rbgRule.ResourceNames) != 1 || rbgRule.ResourceNames[0] != "test-rbg" {
		t.Errorf("planner rolebasedgroups must be limited to the RBG name, got %v", rbgRule.ResourceNames)
	}
	scaleRule := findRule(plannerRole.Rules, "rolebasedgroupscalingadapters/scale")
	if scaleRule == nil {
		t.Fatalf("planner Role must grant scale access, got %+v", plannerRole.Rules)
	}
	if !hasVerb(scaleRule.Verbs, "patch") || len(scaleRule.Verbs) != 1 {
		t.Errorf("planner scale verbs must be exactly patch, got %v", scaleRule.Verbs)
	}
	wantScaleNames := []string{"test-rbg-p-engine", "test-rbg-d-engine"}
	if !reflect.DeepEqual(scaleRule.ResourceNames, wantScaleNames) {
		t.Errorf("planner scale resourceNames = %v, want %v", scaleRule.ResourceNames, wantScaleNames)
	}
	if findRule(plannerRole.Rules, "configmaps") != nil {
		t.Errorf("planner Role must not grant configmaps access, got %+v", plannerRole.Rules)
	}
	if len(plannerRole.OwnerReferences) != 1 || plannerRole.OwnerReferences[0].UID != "ras-uid" {
		t.Errorf("planner Role must be owner-referenced to the AutoScaler, got %+v", plannerRole.OwnerReferences)
	}

	// Profiler Role: RBG get only, configmaps get/create/update on the
	// profiling ConfigMap, and nothing else.
	profilerRole := &rbacv1.Role{}
	if err := cl.Get(context.Background(),
		types.NamespacedName{Name: "test-rbg-profiler", Namespace: "test-ns"}, profilerRole); err != nil {
		t.Fatalf("expected profiler Role to exist, got: %v", err)
	}
	profRBGRule := findRule(profilerRole.Rules, "rolebasedgroups")
	if profRBGRule == nil {
		t.Fatalf("profiler Role must grant rolebasedgroups get, got %+v", profilerRole.Rules)
	}
	if len(profRBGRule.Verbs) != 1 || profRBGRule.Verbs[0] != "get" {
		t.Errorf("profiler rolebasedgroups verbs must be exactly get, got %v", profRBGRule.Verbs)
	}
	if len(profRBGRule.ResourceNames) != 1 || profRBGRule.ResourceNames[0] != "test-rbg" {
		t.Errorf("profiler rolebasedgroups must be limited to the RBG name, got %v", profRBGRule.ResourceNames)
	}
	cmRule := findRule(profilerRole.Rules, "configmaps")
	if cmRule == nil {
		t.Fatalf("profiler Role must grant configmaps access, got %+v", profilerRole.Rules)
	}
	for _, verb := range []string{"get", "create", "update"} {
		if !hasVerb(cmRule.Verbs, verb) {
			t.Errorf("profiler configmaps verbs must include %q, got %v", verb, cmRule.Verbs)
		}
	}
	if hasVerb(cmRule.Verbs, "delete") || hasVerb(cmRule.Verbs, "list") {
		t.Errorf("profiler configmaps verbs must not include delete/list, got %v", cmRule.Verbs)
	}
	if len(cmRule.ResourceNames) != 1 || cmRule.ResourceNames[0] != "test-rbg-profiling" {
		t.Errorf("profiler configmaps must be limited to the profiling ConfigMap, got %v", cmRule.ResourceNames)
	}
	if findRule(profilerRole.Rules, "rolebasedgroupscalingadapters/scale") != nil {
		t.Errorf("profiler Role must not grant scale access, got %+v", profilerRole.Rules)
	}
	if len(profilerRole.OwnerReferences) != 1 || profilerRole.OwnerReferences[0].UID != "ras-uid" {
		t.Errorf("profiler Role must be owner-referenced to the AutoScaler, got %+v", profilerRole.OwnerReferences)
	}

	// RoleBindings wire each SA to its own namespaced Role.
	for _, rbName := range []string{"test-rbg-planner", "test-rbg-profiler"} {
		rb := &rbacv1.RoleBinding{}
		if err := cl.Get(context.Background(),
			types.NamespacedName{Name: rbName, Namespace: "test-ns"}, rb); err != nil {
			t.Fatalf("expected RoleBinding %s to exist, got: %v", rbName, err)
		}
		if rb.RoleRef.Kind != "Role" || rb.RoleRef.Name != rbName {
			t.Errorf("RoleBinding %s must reference Role %s, got %+v", rbName, rbName, rb.RoleRef)
		}
		if len(rb.Subjects) != 1 || rb.Subjects[0].Kind != "ServiceAccount" ||
			rb.Subjects[0].Name != rbName || rb.Subjects[0].Namespace != "test-ns" {
			t.Errorf("RoleBinding %s must bind SA %s in test-ns, got %+v", rbName, rbName, rb.Subjects)
		}
		if len(rb.OwnerReferences) != 1 || rb.OwnerReferences[0].UID != "ras-uid" {
			t.Errorf("RoleBinding %s must be owner-referenced to the AutoScaler, got %+v", rbName, rb.OwnerReferences)
		}
	}

	// No cluster-scoped RBAC may be created.
	crs := &rbacv1.ClusterRoleList{}
	if err := cl.List(context.Background(), crs); err != nil {
		t.Fatalf("failed to list ClusterRoles: %v", err)
	}
	if len(crs.Items) != 0 {
		t.Errorf("expected no ClusterRoles to be created, got %d", len(crs.Items))
	}
	crbs := &rbacv1.ClusterRoleBindingList{}
	if err := cl.List(context.Background(), crbs); err != nil {
		t.Fatalf("failed to list ClusterRoleBindings: %v", err)
	}
	if len(crbs.Items) != 0 {
		t.Errorf("expected no ClusterRoleBindings to be created, got %d", len(crbs.Items))
	}
}

func TestEnsureRBAC_DefaultRoleNames(t *testing.T) {
	scheme := newRBACTestScheme(t)
	ras := newTestRAS()

	cl := fake.NewClientBuilder().WithScheme(scheme).Build()
	r := &AutoScalerReconciler{Client: cl, Scheme: scheme}

	spec := &parsedSpec{prefillRoleName: "prefill", decodeRoleName: "decode"}
	if err := r.ensureRBAC(context.Background(), ras, spec); err != nil {
		t.Fatalf("ensureRBAC failed: %v", err)
	}

	plannerRole := &rbacv1.Role{}
	if err := cl.Get(context.Background(),
		types.NamespacedName{Name: "test-rbg-planner", Namespace: "test-ns"}, plannerRole); err != nil {
		t.Fatalf("expected planner Role to exist, got: %v", err)
	}
	scaleRule := findRule(plannerRole.Rules, "rolebasedgroupscalingadapters/scale")
	if scaleRule == nil {
		t.Fatalf("planner Role must grant scale access, got %+v", plannerRole.Rules)
	}
	want := []string{"test-rbg-prefill", "test-rbg-decode"}
	if !reflect.DeepEqual(scaleRule.ResourceNames, want) {
		t.Errorf("scale resourceNames = %v, want %v", scaleRule.ResourceNames, want)
	}
}

func TestEnsureRBAC_UpdatesResourceNamesWhenRoleNamesChange(t *testing.T) {
	scheme := newRBACTestScheme(t)
	ras := newTestRAS()

	cl := fake.NewClientBuilder().WithScheme(scheme).Build()
	r := &AutoScalerReconciler{Client: cl, Scheme: scheme}

	if err := r.ensureRBAC(context.Background(), ras, &parsedSpec{prefillRoleName: "p1", decodeRoleName: "d1"}); err != nil {
		t.Fatalf("first ensureRBAC failed: %v", err)
	}
	if err := r.ensureRBAC(context.Background(), ras, &parsedSpec{prefillRoleName: "p2", decodeRoleName: "d2"}); err != nil {
		t.Fatalf("second ensureRBAC failed: %v", err)
	}

	plannerRole := &rbacv1.Role{}
	if err := cl.Get(context.Background(),
		types.NamespacedName{Name: "test-rbg-planner", Namespace: "test-ns"}, plannerRole); err != nil {
		t.Fatalf("expected planner Role to exist, got: %v", err)
	}
	scaleRule := findRule(plannerRole.Rules, "rolebasedgroupscalingadapters/scale")
	if scaleRule == nil {
		t.Fatalf("planner Role must grant scale access, got %+v", plannerRole.Rules)
	}
	want := []string{"test-rbg-p2", "test-rbg-d2"}
	if !reflect.DeepEqual(scaleRule.ResourceNames, want) {
		t.Errorf("scale resourceNames = %v, want %v", scaleRule.ResourceNames, want)
	}
}

func TestCreateProfilingJob_UsesProfilerServiceAccount(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := batchv1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add batch scheme: %v", err)
	}
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add core scheme: %v", err)
	}

	ras := newTestRAS()

	cl := fake.NewClientBuilder().WithScheme(scheme).Build()
	r := &AutoScalerReconciler{Client: cl, Scheme: scheme}

	spec := &parsedSpec{
		profilingImage: "ghcr.io/rolebasedgroup/rbg-profiler:latest",
		backend:        "sglang-oai",
	}
	if err := r.createProfilingJob(context.Background(), ras, spec, "test-rbg-profiling", "test-rbg-profiling"); err != nil {
		t.Fatalf("createProfilingJob failed: %v", err)
	}

	created := &batchv1.Job{}
	if err := cl.Get(context.Background(),
		types.NamespacedName{Name: "test-rbg-profiling", Namespace: "test-ns"}, created); err != nil {
		t.Fatalf("failed to get created job: %v", err)
	}
	if got := created.Spec.Template.Spec.ServiceAccountName; got != "test-rbg-profiler" {
		t.Errorf("profiling job must use the dedicated profiler SA, got %q", got)
	}
}

// --- operator ClusterRole escalation-coverage test ---
//
// The Kubernetes API server rejects creating a Role whose rules grant
// permissions the creator does not itself hold (RBAC escalation prevention).
// The operator's ClusterRole (generated from kubebuilder markers into
// config/rbac/role.yaml) must therefore hold every verb+resource it delegates
// via buildPlannerRules/buildProfilerRules.

func TestOperatorClusterRoleCoversDelegatedRules(t *testing.T) {
	data, err := os.ReadFile("../../config/rbac/role.yaml")
	if err != nil {
		t.Fatalf("failed to read generated operator role: %v", err)
	}
	operatorRole := &rbacv1.ClusterRole{}
	if err := yaml.UnmarshalStrict(data, operatorRole); err != nil {
		t.Fatalf("failed to parse config/rbac/role.yaml: %v", err)
	}

	covers := func(rule rbacv1.PolicyRule) error {
		for _, op := range operatorRole.Rules {
			if op.APIGroups[0] != rule.APIGroups[0] {
				continue
			}
			resourceMatch := false
			for _, res := range op.Resources {
				if res == rule.Resources[0] {
					resourceMatch = true
					break
				}
			}
			if !resourceMatch {
				continue
			}
			missing := []string{}
			for _, verb := range rule.Verbs {
				if !hasVerb(op.Verbs, verb) {
					missing = append(missing, verb)
				}
			}
			if len(missing) == 0 {
				return nil
			}
			return fmt.Errorf("operator rule for %s.%s lacks verbs %v (has %v); the API server will reject Role creation via RBAC escalation prevention",
				rule.APIGroups[0], rule.Resources[0], missing, op.Verbs)
		}
		return fmt.Errorf("operator ClusterRole has no rule for %s.%s", rule.APIGroups[0], rule.Resources[0])
	}

	for _, rule := range buildPlannerRules("rbg", "prefill", "decode") {
		if err := covers(rule); err != nil {
			t.Error(err)
		}
	}
	for _, rule := range buildProfilerRules("rbg") {
		if err := covers(rule); err != nil {
			t.Error(err)
		}
	}
}

func TestCleanupClusterResources_NothingToCleanUp(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := batchv1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add batch scheme: %v", err)
	}
	if err := rbacv1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add rbac scheme: %v", err)
	}

	ras := &unstructured.Unstructured{}
	ras.SetName("test-rbg")
	ras.SetNamespace("test-ns")

	cl := fake.NewClientBuilder().
		WithScheme(scheme).
		Build()

	r := &AutoScalerReconciler{Client: cl, Scheme: scheme}

	if err := r.cleanupClusterResources(context.Background(), ras); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
}

func TestBuildPlannerEnv_MetricQueriesJSON(t *testing.T) {
	spec := &parsedSpec{
		prefillRoleName:    "prefill",
		decodeRoleName:     "decode",
		backend:            "sglang-oai",
		modelName:          "Qwen",
		scalingInterval:    180,
		ttft:               200,
		itl:                20,
		loadPredictor:      "arima",
		predictionWindow:   50,
		metricsPort:        9091,
		prometheusEndpoint: "http://prom:9090",
		metricQueriesJSON:  `{"ttft":"sum(rate(custom_ttft[5m]))"}`,
	}

	env := buildPlannerEnv("test-rbg", spec, "ns", 8, 1, 1)

	var got string
	for _, e := range env {
		if e.Name == "METRIC_QUERIES_JSON" {
			got = e.Value
		}
	}
	if got != `{"ttft":"sum(rate(custom_ttft[5m]))"}` {
		t.Errorf("expected METRIC_QUERIES_JSON set, got %q", got)
	}
}

func TestBuildPlannerEnv_NoMetricQueriesJSONWhenEmpty(t *testing.T) {
	spec := &parsedSpec{
		prefillRoleName:    "prefill",
		decodeRoleName:     "decode",
		backend:            "sglang-oai",
		modelName:          "Qwen",
		scalingInterval:    180,
		ttft:               200,
		itl:                20,
		loadPredictor:      "arima",
		predictionWindow:   50,
		metricsPort:        9091,
		prometheusEndpoint: "http://prom:9090",
	}

	env := buildPlannerEnv("test-rbg", spec, "ns", 8, 1, 1)

	for _, e := range env {
		if e.Name == "METRIC_QUERIES_JSON" {
			t.Errorf("did not expect METRIC_QUERIES_JSON when empty, got %q", e.Value)
		}
	}
}

func TestBuildPlannerEnv_MetricEngine(t *testing.T) {
	spec := &parsedSpec{
		prefillRoleName:    "prefill",
		decodeRoleName:     "decode",
		backend:            "sglang-oai",
		modelName:          "Qwen",
		scalingInterval:    180,
		ttft:               200,
		itl:                20,
		loadPredictor:      "arima",
		predictionWindow:   50,
		metricsPort:        9091,
		prometheusEndpoint: "http://prom:9090",
		metricEngine:       "vllm",
	}

	env := buildPlannerEnv("test-rbg", spec, "ns", 8, 1, 1)

	var got string
	for _, e := range env {
		if e.Name == "METRIC_ENGINE" {
			got = e.Value
		}
	}
	if got != "vllm" {
		t.Errorf("expected METRIC_ENGINE vllm, got %q", got)
	}
}

func TestParseRASSpec_MetricsEngineAndQueries(t *testing.T) {
	ras := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"spec": map[string]interface{}{
				"scalingInterval": 180,
				"backend":         "sglang-oai",
				"pattern": map[string]interface{}{
					"pdDisaggregated": map[string]interface{}{
						"prefill": map[string]interface{}{
							"roleName":    "prefill",
							"maxReplicas": int64(10),
							"minReplicas": int64(1),
						},
						"decode": map[string]interface{}{
							"roleName":    "decode",
							"maxReplicas": int64(10),
							"minReplicas": int64(1),
						},
					},
				},
				"implementation": map[string]interface{}{
					"dynamoPlanner": map[string]interface{}{
						"modelName": "Qwen",
						"sla": map[string]interface{}{
							"ttft": 200.0,
							"itl":  20.0,
						},
						"metrics": map[string]interface{}{
							"engine": "vllm",
							"port":   int64(9091),
							"queries": map[string]interface{}{
								"ttft": "sum(rate(vllm:time_to_first_token_seconds_v2_sum[5m]))",
							},
						},
					},
				},
			},
		},
	}

	r := &AutoScalerReconciler{}
	spec, err := r.parseRASSpec(ras)
	if err != nil {
		t.Fatalf("parseRASSpec failed: %v", err)
	}
	if spec.metricEngine != "vllm" {
		t.Errorf("expected metricEngine vllm, got %q", spec.metricEngine)
	}
	if spec.metricQueriesJSON == "" {
		t.Fatal("expected non-empty metricQueriesJSON")
	}
	if !strings.Contains(spec.metricQueriesJSON, "vllm:time_to_first_token_seconds_v2_sum") {
		t.Errorf("expected metricQueriesJSON to contain the overridden ttft, got %q", spec.metricQueriesJSON)
	}
}

func TestParseRASSpec_NoQueries_BackwardCompat(t *testing.T) {
	ras := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"spec": map[string]interface{}{
				"scalingInterval": 180,
				"backend":         "sglang-oai",
				"pattern": map[string]interface{}{
					"pdDisaggregated": map[string]interface{}{
						"prefill": map[string]interface{}{
							"roleName":    "prefill",
							"maxReplicas": int64(10),
							"minReplicas": int64(1),
						},
						"decode": map[string]interface{}{
							"roleName":    "decode",
							"maxReplicas": int64(10),
							"minReplicas": int64(1),
						},
					},
				},
				"implementation": map[string]interface{}{
					"dynamoPlanner": map[string]interface{}{
						"modelName": "Qwen",
						"sla": map[string]interface{}{
							"ttft": 200.0,
							"itl":  20.0,
						},
						"metrics": map[string]interface{}{
							"engine": "sglang",
							"port":   int64(9091),
						},
					},
				},
			},
		},
	}

	r := &AutoScalerReconciler{}
	spec, err := r.parseRASSpec(ras)
	if err != nil {
		t.Fatalf("parseRASSpec failed: %v", err)
	}
	if spec.metricEngine != "sglang" {
		t.Errorf("expected metricEngine sglang, got %q", spec.metricEngine)
	}
	if spec.metricQueriesJSON != "" {
		t.Errorf("expected empty metricQueriesJSON, got %q", spec.metricQueriesJSON)
	}
}

func TestParseRASSpec_MetricQueries_GoldenFixture(t *testing.T) {
	goldenPath := "../../python/planner/tests/fixtures/metric_config_golden.json"
	expected, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("failed to read golden fixture: %v", err)
	}

	ras := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"spec": map[string]interface{}{
				"scalingInterval": 180,
				"backend":         "sglang-oai",
				"pattern": map[string]interface{}{
					"pdDisaggregated": map[string]interface{}{
						"prefill": map[string]interface{}{
							"roleName":    "prefill",
							"maxReplicas": int64(10),
							"minReplicas": int64(1),
						},
						"decode": map[string]interface{}{
							"roleName":    "decode",
							"maxReplicas": int64(10),
							"minReplicas": int64(1),
						},
					},
				},
				"implementation": map[string]interface{}{
					"dynamoPlanner": map[string]interface{}{
						"modelName": "Qwen",
						"sla": map[string]interface{}{
							"ttft": 200.0,
							"itl":  20.0,
						},
						"metrics": map[string]interface{}{
							"engine": "sglang",
							"port":   int64(9091),
							"queries": map[string]interface{}{
								"ttft": "sum(rate(custom_ttft_seconds{modelFilter}[{interval}]))",
								"isl":  "sum(increase(custom_prompt_tokens{modelFilter}[{interval}])) / sum(increase(custom_requests{modelFilter}[{interval}]))",
							},
						},
					},
				},
			},
		},
	}

	r := &AutoScalerReconciler{}
	spec, err := r.parseRASSpec(ras)
	if err != nil {
		t.Fatalf("parseRASSpec failed: %v", err)
	}

	var got, want map[string]interface{}
	if err := json.Unmarshal([]byte(spec.metricQueriesJSON), &got); err != nil {
		t.Fatalf("failed to unmarshal metricQueriesJSON: %v", err)
	}
	if err := json.Unmarshal(expected, &want); err != nil {
		t.Fatalf("failed to unmarshal golden fixture: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("metricQueriesJSON does not match golden fixture\n got: %s\n want: %s", spec.metricQueriesJSON, string(expected))
	}
}

func TestParseRASSpec_ImagePullSecrets(t *testing.T) {
	ras := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"spec": map[string]interface{}{
				"scalingInterval": 180,
				"backend":         "sglang-oai",
				"pattern": map[string]interface{}{
					"pdDisaggregated": map[string]interface{}{
						"prefill": map[string]interface{}{
							"roleName":    "prefill",
							"maxReplicas": int64(10),
							"minReplicas": int64(1),
						},
						"decode": map[string]interface{}{
							"roleName":    "decode",
							"maxReplicas": int64(10),
							"minReplicas": int64(1),
						},
					},
				},
				"implementation": map[string]interface{}{
					"dynamoPlanner": map[string]interface{}{
						"modelName": "Qwen",
						"sla": map[string]interface{}{
							"ttft": 200.0,
							"itl":  20.0,
						},
						"imagePullSecrets": []interface{}{
							map[string]interface{}{"name": "regcred"},
							map[string]interface{}{"name": "ghcr-secret"},
						},
					},
				},
			},
		},
	}

	r := &AutoScalerReconciler{}
	spec, err := r.parseRASSpec(ras)
	if err != nil {
		t.Fatalf("parseRASSpec failed: %v", err)
	}
	if len(spec.imagePullSecrets) != 2 {
		t.Fatalf("expected 2 imagePullSecrets, got %d", len(spec.imagePullSecrets))
	}
	if spec.imagePullSecrets[0].Name != "regcred" {
		t.Errorf("expected first secret 'regcred', got %q", spec.imagePullSecrets[0].Name)
	}
	if spec.imagePullSecrets[1].Name != "ghcr-secret" {
		t.Errorf("expected second secret 'ghcr-secret', got %q", spec.imagePullSecrets[1].Name)
	}
}

func TestParseRASSpec_InputFormat(t *testing.T) {
	ras := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"spec": map[string]interface{}{
				"scalingInterval": 180,
				"backend":         "sglang-oai",
				"pattern": map[string]interface{}{
					"pdDisaggregated": map[string]interface{}{
						"prefill": map[string]interface{}{
							"roleName":    "prefill",
							"maxReplicas": int64(10),
							"minReplicas": int64(1),
						},
						"decode": map[string]interface{}{
							"roleName":    "decode",
							"maxReplicas": int64(10),
							"minReplicas": int64(1),
						},
					},
				},
				"implementation": map[string]interface{}{
					"dynamoPlanner": map[string]interface{}{
						"modelName": "Qwen",
						"profiling": map[string]interface{}{
							"inputFormat": "text",
						},
					},
				},
			},
		},
	}

	r := &AutoScalerReconciler{}
	spec, err := r.parseRASSpec(ras)
	if err != nil {
		t.Fatalf("parseRASSpec failed: %v", err)
	}
	if spec.profilingInputFormat != "text" {
		t.Fatalf("expected inputFormat 'text', got %q", spec.profilingInputFormat)
	}
}

func TestParseRASSpec_InputFormatDefaultsToAuto(t *testing.T) {
	ras := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"spec": map[string]interface{}{
				"pattern": map[string]interface{}{
					"pdDisaggregated": map[string]interface{}{
						"prefill": map[string]interface{}{
							"roleName":    "prefill",
							"maxReplicas": int64(10),
						},
						"decode": map[string]interface{}{
							"roleName":    "decode",
							"maxReplicas": int64(10),
						},
					},
				},
				"implementation": map[string]interface{}{
					"dynamoPlanner": map[string]interface{}{
						"modelName": "Qwen",
					},
				},
			},
		},
	}

	r := &AutoScalerReconciler{}
	spec, err := r.parseRASSpec(ras)
	if err != nil {
		t.Fatalf("parseRASSpec failed: %v", err)
	}
	if spec.profilingInputFormat != "auto" {
		t.Fatalf("expected default inputFormat 'auto', got %q", spec.profilingInputFormat)
	}
}

func TestCreateProfilingJob_InputFormatArgs(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := batchv1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add batch scheme: %v", err)
	}
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add core scheme: %v", err)
	}

	newJob := func(inputFormat string) *batchv1.Job {
		ras := &unstructured.Unstructured{}
		ras.SetName("test-rbg")
		ras.SetNamespace("test-ns")
		ras.SetUID("test-uid")

		cl := fake.NewClientBuilder().WithScheme(scheme).Build()
		r := &AutoScalerReconciler{Client: cl, Scheme: scheme}

		spec := &parsedSpec{
			profilingImage:       "ghcr.io/rolebasedgroup/rbg-profiler:latest",
			backend:              "sglang-oai",
			profilingInputFormat: inputFormat,
		}
		if err := r.createProfilingJob(context.Background(), ras, spec, "test-rbg-profiling", "test-rbg-profiling"); err != nil {
			t.Fatalf("createProfilingJob failed: %v", err)
		}
		created := &batchv1.Job{}
		if err := cl.Get(context.Background(),
			types.NamespacedName{Name: "test-rbg-profiling", Namespace: "test-ns"}, created); err != nil {
			t.Fatalf("failed to get created job: %v", err)
		}
		return created
	}

	containsArg := func(args []string, flag, value string) bool {
		for i, a := range args {
			if a == flag && i+1 < len(args) && args[i+1] == value {
				return true
			}
		}
		return false
	}

	// Explicit inputFormat is forwarded as --input-format.
	job := newJob("text")
	args := job.Spec.Template.Spec.Containers[0].Args
	if !containsArg(args, "--input-format", "text") {
		t.Errorf("expected args to contain '--input-format text', got %v", args)
	}

	// Default "auto" must not add --input-format so the profiler keeps
	// backend-specific defaults.
	job = newJob("auto")
	args = job.Spec.Template.Spec.Containers[0].Args
	for i, a := range args {
		if a == "--input-format" {
			t.Errorf("expected no '--input-format' arg for auto default, got %v (at index %d: %s)", args, i, a)
		}
	}
}

func TestCreateProfilingJob_SetsImagePullSecrets(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := batchv1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add batch scheme: %v", err)
	}
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add core scheme: %v", err)
	}

	ras := &unstructured.Unstructured{}
	ras.SetName("test-rbg")
	ras.SetNamespace("test-ns")
	ras.SetUID("test-uid")

	cl := fake.NewClientBuilder().WithScheme(scheme).Build()
	r := &AutoScalerReconciler{Client: cl, Scheme: scheme}

	spec := &parsedSpec{
		profilingImage: "ghcr.io/rolebasedgroup/rbg-profiler:latest",
		backend:        "sglang-oai",
		imagePullSecrets: []corev1.LocalObjectReference{
			{Name: "regcred"},
		},
	}

	err := r.createProfilingJob(context.Background(), ras, spec, "test-rbg-profiling", "test-rbg-profiling")
	if err != nil {
		t.Fatalf("createProfilingJob failed: %v", err)
	}

	created := &batchv1.Job{}
	if err := cl.Get(context.Background(),
		types.NamespacedName{Name: "test-rbg-profiling", Namespace: "test-ns"}, created); err != nil {
		t.Fatalf("failed to get created job: %v", err)
	}

	if len(created.Spec.Template.Spec.ImagePullSecrets) != 1 {
		t.Fatalf("expected 1 imagePullSecret, got %d", len(created.Spec.Template.Spec.ImagePullSecrets))
	}
	if created.Spec.Template.Spec.ImagePullSecrets[0].Name != "regcred" {
		t.Errorf("expected 'regcred', got %q", created.Spec.Template.Spec.ImagePullSecrets[0].Name)
	}
}

func TestEnsurePlannerDeployment_SetsImagePullSecrets(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := appsv1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add apps scheme: %v", err)
	}
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add core scheme: %v", err)
	}

	ras := &unstructured.Unstructured{}
	ras.SetName("test-rbg")
	ras.SetNamespace("test-ns")
	ras.SetUID("test-uid")

	cl := fake.NewClientBuilder().WithScheme(scheme).Build()
	r := &AutoScalerReconciler{Client: cl, Scheme: scheme}

	spec := &parsedSpec{
		plannerImage: "ghcr.io/rolebasedgroup/rbg-planner:latest",
		imagePullSecrets: []corev1.LocalObjectReference{
			{Name: "regcred"},
		},
	}

	err := r.ensurePlannerDeployment(context.Background(), ras, spec, "test-profiling-cm", 1, 1)
	if err != nil {
		t.Fatalf("ensurePlannerDeployment failed: %v", err)
	}

	created := &appsv1.Deployment{}
	if err := cl.Get(context.Background(),
		types.NamespacedName{Name: "test-rbg-planner", Namespace: "test-ns"}, created); err != nil {
		t.Fatalf("failed to get created deployment: %v", err)
	}

	if len(created.Spec.Template.Spec.ImagePullSecrets) != 1 {
		t.Fatalf("expected 1 imagePullSecret, got %d", len(created.Spec.Template.Spec.ImagePullSecrets))
	}
	if created.Spec.Template.Spec.ImagePullSecrets[0].Name != "regcred" {
		t.Errorf("expected 'regcred', got %q", created.Spec.Template.Spec.ImagePullSecrets[0].Name)
	}
}

func TestParseRASSpec_MetricsBasicAuth(t *testing.T) {
	ras := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"spec": map[string]interface{}{
				"pattern": map[string]interface{}{
					"pdDisaggregated": map[string]interface{}{},
				},
				"implementation": map[string]interface{}{
					"dynamoPlanner": map[string]interface{}{
						"metrics": map[string]interface{}{
							"endpoint": "https://sls.example.com/prometheus/ws",
							"authentication": map[string]interface{}{
								"basicAuth": map[string]interface{}{
									"username": map[string]interface{}{"name": "prom-auth", "key": "user"},
									"password": map[string]interface{}{"name": "prom-auth", "key": "pass"},
								},
							},
						},
					},
				},
			},
		},
	}

	r := &AutoScalerReconciler{PrometheusEndpoint: "http://default:9090"}
	spec, err := r.parseRASSpec(ras)
	if err != nil {
		t.Fatalf("parseRASSpec failed: %v", err)
	}
	if spec.prometheusEndpoint != "https://sls.example.com/prometheus/ws" {
		t.Errorf("expected endpoint override, got %q", spec.prometheusEndpoint)
	}
	if spec.prometheusUsernameSecret == nil || spec.prometheusUsernameSecret.Name != "prom-auth" || spec.prometheusUsernameSecret.Key != "user" {
		t.Errorf("username selector mismatch: %+v", spec.prometheusUsernameSecret)
	}
	if spec.prometheusPasswordSecret == nil || spec.prometheusPasswordSecret.Name != "prom-auth" || spec.prometheusPasswordSecret.Key != "pass" {
		t.Errorf("password selector mismatch: %+v", spec.prometheusPasswordSecret)
	}
	if spec.prometheusTokenSecret != nil {
		t.Errorf("expected no token secret, got %+v", spec.prometheusTokenSecret)
	}

	env := buildPlannerEnv("rbg", spec, "default", 8, 1, 1)
	var user, pass *corev1.EnvVarSource
	for _, e := range env {
		switch e.Name {
		case "PROMETHEUS_USERNAME":
			user = e.ValueFrom
		case "PROMETHEUS_PASSWORD":
			pass = e.ValueFrom
		}
	}
	if user == nil || user.SecretKeyRef == nil || user.SecretKeyRef.Name != "prom-auth" || user.SecretKeyRef.Key != "user" {
		t.Errorf("expected PROMETHEUS_USERNAME secretKeyRef prom-auth/user, got %+v", user)
	}
	if pass == nil || pass.SecretKeyRef == nil || pass.SecretKeyRef.Name != "prom-auth" || pass.SecretKeyRef.Key != "pass" {
		t.Errorf("expected PROMETHEUS_PASSWORD secretKeyRef prom-auth/pass, got %+v", pass)
	}
}

func metricsAuthRAS(me map[string]interface{}) *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"spec": map[string]interface{}{
				"pattern": map[string]interface{}{
					"pdDisaggregated": map[string]interface{}{},
				},
				"implementation": map[string]interface{}{
					"dynamoPlanner": map[string]interface{}{
						"metrics": me,
					},
				},
			},
		},
	}
}

func TestParseRASSpec_MetricsAuthRejectsInvalid(t *testing.T) {
	cases := map[string]map[string]interface{}{
		"basicAuth and bearer together": {
			"authentication": map[string]interface{}{
				"basicAuth": map[string]interface{}{
					"username": map[string]interface{}{"name": "s", "key": "u"},
					"password": map[string]interface{}{"name": "s", "key": "p"},
				},
				"bearer": map[string]interface{}{
					"token": map[string]interface{}{"name": "s", "key": "t"},
				},
			},
		},
		"basicAuth username missing key": {
			"authentication": map[string]interface{}{
				"basicAuth": map[string]interface{}{
					"username": map[string]interface{}{"name": "s"},
					"password": map[string]interface{}{"name": "s", "key": "p"},
				},
			},
		},
		"basicAuth password missing": {
			"authentication": map[string]interface{}{
				"basicAuth": map[string]interface{}{
					"username": map[string]interface{}{"name": "s", "key": "u"},
				},
			},
		},
		"bearer token missing key": {
			"authentication": map[string]interface{}{
				"bearer": map[string]interface{}{
					"token": map[string]interface{}{"name": "s"},
				},
			},
		},
	}
	for name, me := range cases {
		r := &AutoScalerReconciler{}
		if _, err := r.parseRASSpec(metricsAuthRAS(me)); err == nil {
			t.Errorf("%s: expected error, got nil", name)
		}
	}
}

func TestParseRASSpec_MetricsBearerToken(t *testing.T) {
	me := map[string]interface{}{
		"authentication": map[string]interface{}{
			"bearer": map[string]interface{}{
				"token": map[string]interface{}{"name": "prom-auth", "key": "token"},
			},
		},
	}
	r := &AutoScalerReconciler{PrometheusEndpoint: "http://default:9090"}
	spec, err := r.parseRASSpec(metricsAuthRAS(me))
	if err != nil {
		t.Fatalf("parseRASSpec failed: %v", err)
	}
	if spec.prometheusTokenSecret == nil || spec.prometheusTokenSecret.Name != "prom-auth" || spec.prometheusTokenSecret.Key != "token" {
		t.Fatalf("token selector mismatch: %+v", spec.prometheusTokenSecret)
	}
	if spec.prometheusUsernameSecret != nil || spec.prometheusPasswordSecret != nil {
		t.Fatalf("expected no basic auth selectors, got %+v/%+v", spec.prometheusUsernameSecret, spec.prometheusPasswordSecret)
	}

	env := buildPlannerEnv("rbg", spec, "default", 8, 1, 1)
	var token *corev1.EnvVarSource
	hasBasic := false
	for _, e := range env {
		switch e.Name {
		case "PROMETHEUS_TOKEN":
			token = e.ValueFrom
		case "PROMETHEUS_USERNAME", "PROMETHEUS_PASSWORD":
			hasBasic = true
		}
	}
	if token == nil || token.SecretKeyRef == nil || token.SecretKeyRef.Name != "prom-auth" || token.SecretKeyRef.Key != "token" {
		t.Errorf("expected PROMETHEUS_TOKEN secretKeyRef prom-auth/token, got %+v", token)
	}
	if hasBasic {
		t.Error("bearer auth must not inject PROMETHEUS_USERNAME/PASSWORD")
	}
}

func TestShouldShortCircuitFailed(t *testing.T) {
	tests := []struct {
		name     string
		phase    string
		lastFail int64
		gen      int64
		want     bool
	}{
		{"not failed", "Initializing", 0, 1, false},
		{"failed no spec change", "Failed", 2, 2, true},
		{"failed gen behind", "Failed", 2, 1, true},
		{"failed gen advanced", "Failed", 2, 3, false},
		{"failed no lastFailGen recorded", "Failed", 0, 1, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ras := &unstructured.Unstructured{}
			ras.SetGeneration(tt.gen)
			ras.Object["status"] = map[string]interface{}{
				"phase":                          tt.phase,
				"lastProfilingFailureGeneration": tt.lastFail,
			}
			if got := shouldShortCircuitFailed(ras); got != tt.want {
				t.Errorf("shouldShortCircuitFailed() = %v, want %v", got, tt.want)
			}
		})
	}
}
