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
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/davidkleiven/goselect/featselect"
	"github.com/spf13/cobra"
)

// sasearchCmd represents the sasearch command
var sasearchCmd = &cobra.Command{
	Use:   "sasearch",
	Short: "Performs model selection by minimize AICc",
	Long: `The space of possible models are explored via simmulated annealing (SA). Features are included/exluded and the
corresponding AICc vlaue evaluated. The SA algorithm then minimized AICc.

Example:
goselect sasearch -csv mydatafile.csv -target -1 -out result.json -sweeps 40
	`,
	Run: func(cmd *cobra.Command, args []string) {
		saCsv, err := cmd.Flags().GetString("csv")
		if err != nil {
			log.Print(err)
			return
		}

		saTarget, _ := cmd.Flags().GetInt("target")
		saOut, _ := cmd.Flags().GetString("out")
		saSweeps, _ := cmd.Flags().GetInt("sweeps")

		saSearch(saCsv, saTarget, saOut, saSweeps)
	},
}

func init() {
	rootCmd.AddCommand(sasearchCmd)

	sasearchCmd.Flags().String("csv", "", "CSV file with data")
	sasearchCmd.Flags().Int("target", -1, "Column where the target values are placed. If negative it is counted from the last column.")
	sasearchCmd.Flags().String("out", "saSearch.json", "JSON file where the final result will be stored")
	sasearchCmd.Flags().Int("sweeps", 100, "Number of sweeps per temperature")
}

func saSearch(csvfile string, targetCol int, out string, sweeps int) {
	rand.Seed(time.Now().UTC().UnixNano())
	dset := featselect.ReadCSV(csvfile, targetCol)
	res := featselect.SelectModelSA(dset.X, dset.Y, sweeps, featselect.Aicc)
	file, _ := os.Open(out)
	defer file.Close()

	highscoreJSON, _ := json.Marshal(res.Scores)
	ioutil.WriteFile(out, highscoreJSON, 0644)

	fmt.Printf("SA highscore list written to %s", out)
}
