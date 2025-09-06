
import csv
import sys
import os
import glob
from pathlib import Path


def extract_fields_from_log(file_path):

    with open(file_path, 'r') as infile:
        for line in infile:
            if line.startswith('#fields'):

                fields = line.replace('#fields', '').strip().split('\t')
                # Clean up field names
                fields = [field.strip() for field in fields if field.strip()]
                return fields
            elif not line.startswith('#') and line.strip():

                num_fields = len(line.strip().split('\t'))
                return [f'field_{i + 1}' for i in range(num_fields)]
    return []


def process_conn_log_file(file_path, fields):

    data_rows = []

    with open(file_path, 'r') as infile:

        data_started = False
        for line in infile:

            if line.startswith('#fields'):
                data_started = True
                continue
            elif line.startswith('#'):
                if not data_started:
                    continue
            elif not data_started:
                data_started = True


            if data_started and line.strip() and not line.startswith('#'):
                # Split by tabs and clean up
                row_data = line.strip().split('\t')


                while len(row_data) < len(fields) - 2:
                    row_data.append('')


                if len(row_data) > len(fields) - 2:
                    row_data = row_data[:len(fields) - 2]


                row_data.extend(['1', 'injection'])

                data_rows.append(row_data)

    return data_rows


def convert_all_conn_logs_to_csv(output_file="combined_log_files.csv"):




    current_dir = Path('.')


    log_files = glob.glob('*.log')


    import re
    def natural_sort_key(text):
        return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', text)]

    log_files = sorted(log_files, key=natural_sort_key)

    if not log_files:
        print("No .log files found in the current directory")
        return False

    print(f"Found {len(log_files)} .log file(s):")
    for f in log_files:
        print(f"  - {f}")


    fields = None
    for log_file in log_files:
        try:
            fields = extract_fields_from_log(log_file)
            if fields:
                break
        except Exception as e:
            print(f"Warning: Could not read {log_file}: {e}")
            continue

    if not fields:
        print("Error: Could not determine field structure from any log file")
        return False

    # Add the new columns
    fields.extend(['label', 'type'])

    print(f"\nField structure detected ({len(fields)} columns):")
    for i, field in enumerate(fields, 1):
        print(f"  {i:2d}. {field}")

    # Process all files and combine data
    all_rows = []
    total_rows = 0

    for log_file in log_files:
        try:
            print(f"\nProcessing {log_file}...")
            rows = process_conn_log_file(log_file, fields)
            all_rows.extend(rows)
            print(f"  Added {len(rows)} rows")
            total_rows += len(rows)
        except Exception as e:
            print(f"Error processing {log_file}: {e}")
            continue

    if not all_rows:
        print("No data rows found in any log files")
        return False


    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)


            writer.writerow(fields)


            writer.writerows(all_rows)

        print(f"\n✅ Successfully created {output_file}")
        print(f"📊 Total rows processed: {total_rows}")
        print(f"📁 Files combined: {len(log_files)}")
        print(f"📋 Columns: {len(fields)} (including 'label' and 'type')")

        return True

    except Exception as e:
        print(f"Error writing CSV file: {e}")
        return False


def main():
    print("🔄 Log Files to CSV Converter - Batch Processing")
    print("=" * 50)


    output_file = "combined_log_files.csv"
    if len(sys.argv) > 1:
        output_file = sys.argv[1]

    print(f"Output file: {output_file}")
    print(f"Working directory: {os.getcwd()}")

    try:
        success = convert_all_conn_logs_to_csv(output_file)
        if success:
            print(f"\n🎉 Conversion completed successfully!")
            print(f"📄 Output saved as: {output_file}")
        else:
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()