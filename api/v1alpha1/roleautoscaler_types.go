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

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// AutoScalerSpec defines the desired state of AutoScaler.
type AutoScalerSpec struct {
	// ScalingInterval is the number of seconds between scaling decisions.
	// +optional
	// +kubebuilder:default=180
	ScalingInterval int `json:"scalingInterval,omitempty"`

	// Pattern defines the scaling pattern and per-role configuration.
	Pattern Pattern `json:"pattern"`

	// Implementation defines the scaling engine and its configuration.
	Implementation Implementation `json:"implementation"`
}

// Pattern is a discriminated union of scaling patterns.
// Exactly one field must be set.
type Pattern struct {
	// PDDisaggregated configures Prefill/Decode disaggregated scaling.
	// +optional
	PDDisaggregated *PDDisaggregatedPattern `json:"PDDisaggregated,omitempty"`

	// Unified configures unified (non-disaggregated) scaling.
	// Reserved for future use.
	// +optional
	Unified *UnifiedPattern `json:"Unified,omitempty"`
}

// PDDisaggregatedPattern defines scaling for Prefill/Decode disaggregated roles.
type PDDisaggregatedPattern struct {
	// Prefill defines the prefill role scaling configuration.
	Prefill RoleScalingConfig `json:"prefill"`

	// Decode defines the decode role scaling configuration.
	Decode RoleScalingConfig `json:"decode"`
}

// RoleScalingConfig defines scaling bounds for a single role.
type RoleScalingConfig struct {
	// RoleName is the name of the role in the RoleBasedGroup.
	// +kubebuilder:default="prefill"
	RoleName string `json:"roleName"`

	// MaxReplicas is the maximum number of replicas for this role.
	// +kubebuilder:validation:Minimum=1
	MaxReplicas int32 `json:"maxReplicas"`

	// MinReplicas is the minimum number of replicas for this role.
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:default=1
	MinReplicas int32 `json:"minReplicas"`
}

// UnifiedPattern defines scaling for unified (non-disaggregated) inference.
// Reserved for future use — not yet implemented.
type UnifiedPattern struct {
	// RoleName is the name of the role in the RoleBasedGroup.
	RoleName string `json:"roleName"`

	// MaxReplicas is the maximum number of replicas.
	// +kubebuilder:validation:Minimum=1
	MaxReplicas int32 `json:"maxReplicas"`

	// MinReplicas is the minimum number of replicas.
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:default=1
	MinReplicas int32 `json:"minReplicas"`
}

// Implementation is a discriminated union of scaling engines.
// Exactly one field must be set.
type Implementation struct {
	// DynamoPlanner configures the Dynamo-derived SLA-based planner.
	// +optional
	DynamoPlanner *DynamoPlannerConfig `json:"DynamoPlanner,omitempty"`
}

// DynamoPlannerConfig defines configuration for the Dynamo planner engine.
type DynamoPlannerConfig struct {
	// ModelName is the model name used for Prometheus label filtering.
	// +optional
	ModelName string `json:"modelName,omitempty"`

	// TTFT is the target Time to First Token SLA in milliseconds.
	// +optional
	// +kubebuilder:default=500
	TTFT float64 `json:"ttft,omitempty"`

	// ITL is the target Inter-Token Latency SLA in milliseconds.
	// +optional
	// +kubebuilder:default=50
	ITL float64 `json:"itl,omitempty"`

	// LoadPredictor selects the load prediction algorithm.
	// +optional
	// +kubebuilder:default="arima"
	// +kubebuilder:validation:Enum=arima;constant;prophet
	LoadPredictor string `json:"loadPredictor,omitempty"`

	// PredictionWindow is the number of data points in the predictor window.
	// +optional
	// +kubebuilder:default=50
	PredictionWindow int `json:"predictionWindow,omitempty"`

	// NoCorrection disables SLA correction factors.
	// +optional
	NoCorrection bool `json:"noCorrection,omitempty"`

	// DryRun enables observe-only mode without actual scaling.
	// +optional
	DryRun bool `json:"dryRun,omitempty"`

	// Profiling configures automatic SLA profiling.
	// +optional
	Profiling *ProfilingConfig `json:"profiling,omitempty"`

	// MetricsEndpoint configures the metrics collection endpoint.
	// +optional
	MetricsEndpoint *MetricsEndpointConfig `json:"metricsEndpoint,omitempty"`
}

// ProfilingConfig defines profiling Job configuration.
type ProfilingConfig struct {
	// Image is the container image for the profiling tool.
	// If empty, a default image is used.
	// +optional
	Image string `json:"image,omitempty"`
}

// MetricsEndpointConfig configures metrics collection.
type MetricsEndpointConfig struct {
	// MetricSource selects the metric source type.
	// +optional
	// +kubebuilder:default="sglang"
	// +kubebuilder:validation:Enum=sglang;vllm;patio
	MetricSource string `json:"metricSource,omitempty"`

	// Port is the port for planner's own Prometheus metrics exposition.
	// +optional
	// +kubebuilder:default=9091
	Port int `json:"port,omitempty"`
}

// AutoScalerPhase represents the current phase of an AutoScaler.
// +kubebuilder:validation:Enum=Pending;Initializing;Ready;Failed
type AutoScalerPhase string

const (
	PhasePending      AutoScalerPhase = "Pending"
	PhaseInitializing AutoScalerPhase = "Initializing"
	PhaseReady        AutoScalerPhase = "Ready"
	PhaseFailed       AutoScalerPhase = "Failed"
)

// AutoScalerStatus defines the observed state of AutoScaler.
type AutoScalerStatus struct {
	// Phase indicates the current phase of the AutoScaler.
	// +optional
	Phase AutoScalerPhase `json:"phase,omitempty"`

	// Conditions represent the latest available observations of the AutoScaler's state.
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// PrefillReplicas is the current prefill replica count.
	// +optional
	PrefillReplicas *int32 `json:"prefillReplicas,omitempty"`

	// DecodeReplicas is the current decode replica count.
	// +optional
	DecodeReplicas *int32 `json:"decodeReplicas,omitempty"`

	// ProfilingConfigMap is the name of the ConfigMap containing profiling data.
	// +optional
	ProfilingConfigMap string `json:"profilingConfigMap,omitempty"`

	// PlannerDeployment is the name of the planner Deployment.
	// +optional
	PlannerDeployment string `json:"plannerDeployment,omitempty"`

	// LastTransitionTime is the last time the phase transitioned.
	// +optional
	LastTransitionTime *metav1.Time `json:"lastTransitionTime,omitempty"`
}

// +genclient
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:storageversion
// +kubebuilder:resource:shortName={as}
// +kubebuilder:printcolumn:name="PHASE",type="string",JSONPath=".status.phase",description="Current phase"
// +kubebuilder:printcolumn:name="PREFILL",type="integer",JSONPath=".status.prefillReplicas",description="Prefill replicas"
// +kubebuilder:printcolumn:name="DECODE",type="integer",JSONPath=".status.decodeReplicas",description="Decode replicas"
// +kubebuilder:printcolumn:name="AGE",type="date",JSONPath=".metadata.creationTimestamp"

// AutoScaler is the Schema for the autoscalers API.
// An AutoScaler automatically scales roles within a RoleBasedGroup
// to meet SLA targets. The AutoScaler name must match the target
// RoleBasedGroup name in the same namespace.
type AutoScaler struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   AutoScalerSpec   `json:"spec,omitempty"`
	Status AutoScalerStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// AutoScalerList contains a list of AutoScaler.
type AutoScalerList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []AutoScaler `json:"items"`
}

func init() {
	SchemeBuilder.Register(&AutoScaler{}, &AutoScalerList{})
}
