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

func TestHandleProfiling_FailedJobAlreadyDeletingSkipsDelete(t *testing.T) {
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

	_, _, err := r.handleProfiling(context.Background(), ras, &parsedSpec{})
	if err == nil {
		t.Fatal("expected error for failed job, got nil")
	}
	if !strings.Contains(err.Error(), "failed") {
		t.Errorf("expected error to contain 'failed', got: %v", err)
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
	crb := &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-rbg-planner-binding",
		},
	}

	cl := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(job, crb).
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

	// CRB should still exist — cleanup must not proceed past the Job.
	if err := cl.Get(context.Background(),
		types.NamespacedName{Name: "test-rbg-planner-binding"},
		&rbacv1.ClusterRoleBinding{}); err != nil {
		t.Errorf("expected CRB to still exist, got error: %v", err)
	}
}

func TestCleanupClusterResources_JobGoneDeletesCRB(t *testing.T) {
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

	crb := &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-rbg-planner-binding",
		},
	}

	cl := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(crb).
		Build()

	r := &AutoScalerReconciler{Client: cl, Scheme: scheme}

	if err := r.cleanupClusterResources(context.Background(), ras); err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}

	// CRB should be deleted.
	err := cl.Get(context.Background(),
		types.NamespacedName{Name: "test-rbg-planner-binding"},
		&rbacv1.ClusterRoleBinding{})
	if !apierrors.IsNotFound(err) {
		t.Errorf("expected CRB to be deleted, got error: %v", err)
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
