import json
import pandas
import os
import glob

# Read direct FourierFT evaluation results (without soft inference)
SCALE = str(0.1)
writer = pandas.ExcelWriter(f"direct_fourierft_results_scale_{SCALE}.xlsx", engine='xlsxwriter')
sheet_name = "direct_unlearn"
sheet = writer.book.add_worksheet(sheet_name)

# Headers
headers = ["SEED", "LABEL_K", "Stage", "SU", "DU", "RD", "CommonQA", "OpenBookQA"]
for col, header in enumerate(headers):
    sheet.write(0, col, header)

row = 1
print("=== Direct FourierFT Results (No Soft Inference) ===")
print("SU = Specific Unlearning (train), DU = Domain Unlearning (test)")
print("RD = Retain Data, CommonQA = General Knowledge, OpenBookQA = General Knowledge")
print("=" * 70)

for LABEL_K in ["force"]:
    for SEED in [0, 1, 2]:
        OUTPUT_1 = f"./SCALE_{SCALE}_seed_{SEED}_o_unlearn_lora_{LABEL_K}_checkpoints_5/lora_{LABEL_K}_random"
        TYPE = ""
        results = []
        
        for UNLEAN_D in ["biology", "physics", "chemistry"]:
            OUTPUT_1 += f"_{UNLEAN_D}_{LABEL_K}"
            TYPE += f"_{UNLEAN_D}"
            
            adapter_name = os.path.basename(OUTPUT_1)
            
            # Look for direct evaluation result files
            try:
                # RD (Retain Data) - should be high
                rd_pattern = f"./data/scienceqa_RD_5/scienceqa_not{TYPE}_test_RD_{adapter_name}_direct_fourierft.json"
                rd_files = glob.glob(rd_pattern)
                if rd_files:
                    with open(rd_files[0], 'r') as f:
                        RD = json.load(f)['accuracy']
                else:
                    print(f"Warning: RD file not found: {rd_pattern}")
                    RD = 0.0

                # SU (Specific Unlearning - train) - should be low
                su_pattern = f"./data/scienceqa_SD_5/scienceqa{TYPE}_train_SD_{adapter_name}_direct_fourierft.json"
                su_files = glob.glob(su_pattern)
                if su_files:
                    with open(su_files[0], 'r') as f:
                        SU = json.load(f)['accuracy']
                else:
                    print(f"Warning: SU file not found: {su_pattern}")
                    SU = 0.0

                # DU (Domain Unlearning - test) - should be low
                du_pattern = f"./data/scienceqa_SD_5/scienceqa{TYPE}_test_SD_{adapter_name}_direct_fourierft.json"
                du_files = glob.glob(du_pattern)
                if du_files:
                    with open(du_files[0], 'r') as f:
                        DU = json.load(f)['accuracy']
                else:
                    print(f"Warning: DU file not found: {du_pattern}")
                    DU = 0.0

                # CommonQA - should be high
                cqa_pattern = f"./data/commonqa/commonqa_test_{adapter_name}_direct_fourierft.json"
                cqa_files = glob.glob(cqa_pattern)
                if cqa_files:
                    with open(cqa_files[0], 'r') as f:
                        CQA = json.load(f)['accuracy']
                else:
                    print(f"Warning: CommonQA file not found: {cqa_pattern}")
                    CQA = 0.0

                # OpenBookQA - should be high
                oqa_pattern = f"./data/openbookqa/openbookqa_test_{adapter_name}_direct_fourierft.json"
                oqa_files = glob.glob(oqa_pattern)
                if oqa_files:
                    with open(oqa_files[0], 'r') as f:
                        OQA = json.load(f)['accuracy']
                else:
                    print(f"Warning: OpenBookQA file not found: {oqa_pattern}")
                    OQA = 0.0

                print(f"SEED: {SEED:2d} | LABEL_K: {LABEL_K:5s} | Stage: {TYPE:20s}")
                print(f"  SU: {SU:.4f} | DU: {DU:.4f} | RD: {RD:.4f} | CommonQA: {CQA:.4f} | OpenBookQA: {OQA:.4f}")
                print(f"  CSV: {SU:.4f},{DU:.4f},{RD:.4f},{CQA:.4f},{OQA:.4f}")
                
                # Write to Excel
                sheet.write(row, 0, f"SEED: {SEED}")
                sheet.write(row, 1, f"LABEL_K: {LABEL_K}")
                sheet.write(row, 2, f"Stage: {TYPE}")
                sheet.write(row, 3, SU)
                sheet.write(row, 4, DU)
                sheet.write(row, 5, RD)
                sheet.write(row, 6, CQA)
                sheet.write(row, 7, OQA)
                
                results += [SU, DU, RD, CQA, OQA, ""]
                
            except Exception as e:
                print(f"Error processing SEED={SEED}, TYPE={TYPE}: {e}")
                results += [0.0, 0.0, 0.0, 0.0, 0.0, ""]
            
            row += 1
            print("-" * 70)

print("\n=== Summary ===")
print("Perfect unlearning would show:")
print("  - SU (Specific Unlearning): LOW (close to 0)")
print("  - DU (Domain Unlearning): LOW (close to 0)")  
print("  - RD (Retain Data): HIGH (close to original performance)")
print("  - CommonQA/OpenBookQA: HIGH (maintained general capability)")
print(f"\nResults saved to: direct_fourierft_results_scale_{SCALE}.xlsx")

writer.close()


