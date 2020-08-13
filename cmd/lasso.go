/*
Copyright © 2020 NAME HERE <EMAIL ADDRESS>

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
package cmd

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/davidkleiven/goselect/featselect"
	"github.com/spf13/cobra"
	"gonum.org/v1/gonum/mat"
)

// lassoCmd represents the lasso command
var lassoCmd = &cobra.Command{
	Use:   "lasso",
	Short: "Performs Lasso fitting",
	Long:  `Performs Lasso fitting`,
	Run: func(cmd *cobra.Command, args []string) {
		lassoCsv, _ := cmd.Flags().GetString("csv")
		if lassoCsv == "" {
			fmt.Printf("No CSV file given!\n")
			return
		}

		target, _ := cmd.Flags().GetInt("target")
		out, _ := cmd.Flags().GetString("out")
		lmin, _ := cmd.Flags().GetFloat64("lmin")
		lmax, _ := cmd.Flags().GetFloat64("lmax")
		num, _ := cmd.Flags().GetInt("num")
		ltype, _ := cmd.Flags().GetString("type")
		cov, _ := cmd.Flags().GetString("cov")
		tol, _ := cmd.Flags().GetFloat64("tol")

		lassoFit(lassoCsv, target, out, lmin, lmax, num, ltype, cov, tol)
	},
}

func init() {
	rootCmd.AddCommand(lassoCmd)

	lassoCmd.Flags().String("csv", "", "CSV file with data")
	lassoCmd.Flags().Int("target", -1, "Target column, if negative the column is counted from the end")
	lassoCmd.Flags().String("out", "lasso.json", "JSON file where the output will be stored")
	lassoCmd.Flags().Float64("lmin", 1e-10, "Minimum value of the regularization parameter")
	lassoCmd.Flags().Float64("lmax", 1.0, "Maximum value of the regularization parameter")
	lassoCmd.Flags().Int("num", 50, "Number of regularization (only with coordinate descent)")
	lassoCmd.Flags().String("type", "lars", "Algorithm lars or cd")
	lassoCmd.Flags().String("cov", "empirical", "Estimator for covariance matrix")
	lassoCmd.Flags().Float64("tol", 1e-4, "Tolerance in LASSO coordinate descent")

}

func lassoFit(csvfile string, targetCol int, out string, lambMin float64, lambMax float64, num int, lassoType string, covType string, tol float64) {
	dset := featselect.ReadCSV(csvfile, targetCol)
	y := make([]float64, len(dset.Y))
	copy(y, dset.Y)
	normDset := featselect.NewNormalizedData(mat.DenseCopyOf(dset.X), y)

	larspath := []*featselect.LassoLarsNode{}
	if lassoType == "lars" {
		var estimator featselect.MorsePenroseCD
		larspath = featselect.LassoLars(normDset, lambMin, &estimator)
	} else if lassoType == "cd" {
		var cov featselect.CovMat
		var corr featselect.PureLasso
		if covType == "empirical" {
			var emp featselect.Empirical
			cov = &emp
		} else if covType == "identity" {
			var id featselect.Identity
			cov = &id
		} else if covType == "threshold" {
			cov = featselect.NewSparseThreshold(normDset.X)
		} else {
			fmt.Printf("Unknown covariance type %s\n", covType)
			return
		}
		lambs := featselect.Logspace(lambMin, lambMax, num)
		larspath = featselect.LassoCrdDescPath(normDset, cov, lambs, 100000, tol, &corr)
	}

	featselect.Path2Unnormalized(normDset, larspath)
	fmt.Printf("LASSO-LARS solution finished. Number of nodes in path %d.\n", len(larspath))

	var path featselect.LassoLarsPath
	path.Dset = dset
	path.LassoLarsNodes = larspath

	aicc := path.GetCriteria(featselect.Aicc)
	bic := path.GetCriteria(featselect.Bic)
	path.Aicc = aicc
	path.Bic = bic
	featselect.PrintHighscore(&path, aicc, bic, 20)

	js, err := json.Marshal(path)

	if err != nil {
		fmt.Printf("Error: %s", err)
		return
	}

	file, _ := os.Create(out)
	defer file.Close()

	ioutil.WriteFile(out, js, 0644)
	fmt.Printf("LASSO-LARS results written to %s\n", out)
}
