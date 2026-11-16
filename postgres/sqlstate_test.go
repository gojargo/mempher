package postgres

import (
	"regexp"
	"slices"
	"strings"
	"testing"
)

// TestAPIErrorCodesAreMapped keeps the two halves of the error contract in
// step: every SQLSTATE the migrations raise or document has a sentinel here, and
// every sentinel here is documented where SQL callers will read it.
func TestAPIErrorCodesAreMapped(t *testing.T) {
	t.Parallel()

	migrations, err := loadMigrations()
	if err != nil {
		t.Fatalf("loadMigrations: %v", err)
	}
	code := regexp.MustCompile(`\bMP\d{3}\b`)
	used := map[string]string{}
	var contract string
	for _, m := range migrations {
		for _, c := range code.FindAllString(m.source, -1) {
			used[c] = m.filename()
		}
		if m.name == "init" {
			contract = m.source
		}
	}
	if contract == "" {
		t.Fatal("0001_init.sql, which documents the contract, is not embedded")
	}
	for c, file := range used {
		if _, ok := apiErrors[c]; !ok {
			t.Errorf("%s raises %s, which apiErrors does not map", file, c)
		}
	}
	for _, c := range slices.Sorted(func(yield func(string) bool) {
		for k := range apiErrors {
			if !yield(k) {
				return
			}
		}
	}) {
		if !strings.Contains(contract, "--     "+c) && !strings.Contains(contract, "  "+c+" ") {
			t.Errorf("apiErrors maps %s, which 0001_init.sql does not document", c)
		}
	}
}
