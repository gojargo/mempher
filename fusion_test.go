package mempher

import (
	"errors"
	"testing"
)

func TestFusionOptionsResolve(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		opts    FusionOptions
		wantK   float64
		wantErr error
	}{
		{name: "zero K takes the default", opts: FusionOptions{}, wantK: DefaultFusionK},
		{name: "explicit K is kept", opts: FusionOptions{K: 1}, wantK: 1},
		{name: "negative K is rejected", opts: FusionOptions{K: -1}, wantErr: ErrInvalidConfig},
		{
			name:    "negative weight is rejected",
			opts:    FusionOptions{Weights: map[Channel]float64{ChannelLexical: -0.5}},
			wantErr: ErrInvalidConfig,
		},
		{
			name:    "unknown channel is rejected",
			opts:    FusionOptions{Weights: map[Channel]float64{Channel("telepathy"): 1}},
			wantErr: ErrInvalidChannel,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			got, err := tc.opts.resolve()
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("err = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got.K != tc.wantK {
				t.Errorf("K = %v, want %v", got.K, tc.wantK)
			}
		})
	}
}

// TestFusionOptionsResolveDetachesWeights guards against a caller mutating a
// live Memory by holding on to the map it passed to New.
func TestFusionOptionsResolveDetachesWeights(t *testing.T) {
	t.Parallel()

	weights := map[Channel]float64{ChannelLexical: 2}
	resolved, err := FusionOptions{Weights: weights}.resolve()
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	weights[ChannelLexical] = 999

	if got := resolved.Weights[ChannelLexical]; got != 2 {
		t.Errorf("weight = %v after mutating the caller's map, want 2", got)
	}
	if got, named := resolved.Weights[ChannelSemantic]; named {
		t.Errorf("unweighted channel resolved to %v, want it left unnamed, which means 1", got)
	}
}

// TestDefaultLexicalWeightIsApplied pins the calibration itself, which the
// tests above deliberately opt out of.
//
// It is a policy rather than arithmetic, and it exists because measurement said
// so: at equal weight, fusing the two channels scored below the semantic
// channel used alone. If this value ever changes it should change because a
// later run of the eval subpackage said to, and this is the test that makes
// that a deliberate act.
func TestDefaultLexicalWeightIsApplied(t *testing.T) {
	t.Parallel()

	resolved, err := FusionOptions{}.resolve()
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if got := resolved.Weights[ChannelLexical]; got != DefaultLexicalWeight {
		t.Errorf("lexical weight = %v, want %v", got, DefaultLexicalWeight)
	}
	// Only the lexical channel is calibrated; everything else is unnamed, which
	// the database reads as a weight of 1.
	if got, named := resolved.Weights[ChannelSemantic]; named {
		t.Errorf("semantic weight resolved to %v, want it left unnamed, which means 1", got)
	}

	t.Run("naming it overrides the default", func(t *testing.T) {
		for _, want := range []float64{0, 1, 3} {
			resolved, err := FusionOptions{
				Weights: map[Channel]float64{ChannelLexical: want},
			}.resolve()
			if err != nil {
				t.Fatalf("resolve: %v", err)
			}
			if got := resolved.Weights[ChannelLexical]; got != want {
				t.Errorf("lexical weight = %v, want the %v that was asked for", got, want)
			}
		}
	})
}
