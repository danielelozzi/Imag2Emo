import os
import json

def generate_data_file(root_dir='results', output_file='embedded_data.js'):
    """
    Scans a directory for summary_metrics_per_classe.csv files and generates 
    a JavaScript file containing their content.

    Args:
        root_dir (str): The root directory to scan (should be 'results').
        output_file (str): The name of the output JavaScript file.
    """
    all_reports_data = {}
    print(f"[*] Starting scan in directory: '{root_dir}'")

    if not os.path.isdir(root_dir):
        print(f"[!] Error: Directory '{root_dir}' not found.")
        print("[!] Please run this script from the same directory that contains the 'results' folder.")
        return

    # Walk through the directory structure
    for dirpath, _, filenames in os.walk(root_dir):
        # We are looking for the 'scaling_ON' directories which contain the summaries
        if os.path.basename(dirpath) == 'scaling_ON':
            per_classe_path = os.path.join(dirpath, 'summary_metrics_per_classe.csv')

            if os.path.exists(per_classe_path):
                # Create a key based on the relative path, compatible with the website's logic
                # e.g., DEAP/PUBLIC/arousal_pubblica/k_simple/EEGNetv4/scaling_ON/
                key = os.path.relpath(dirpath, root_dir).replace('\\', '/') + '/'
                
                print(f"  [+] Found report for: {key}")

                # Read the content of summary_metrics_per_classe.csv
                with open(per_classe_path, 'r', encoding='utf-8') as f:
                    per_classe_content = f.read()

                # Store only the content of summary_metrics_per_classe.csv
                all_reports_data[key] = per_classe_content

    # Write the collected data to the output JavaScript file
    with open(output_file, 'w', encoding='utf-8') as f:
        # Use json.dumps for proper string escaping
        js_content = f"const allReportsData = {json.dumps(all_reports_data, indent=4)};"
        f.write(js_content)

    print(f"\n[*] Success! Data has been written to '{output_file}'")
    print(f"[*] Found data for {len(all_reports_data)} combinations.")

if __name__ == '__main__':
    generate_data_file()
