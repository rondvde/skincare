import pandas as pd
import anthropic
import json
import os
from datetime import datetime

client = anthropic.Anthropic(api_key="INSERT_KEY")

def check_and_download():
    # Reading previous Batch id
    try:
        with open("batch_id.txt", "r") as f:
            batch_id = f.read().strip()
    except FileNotFoundError:
        print("Error: Couldn't find 'batch_id.txt'")
        return

    print(f"Checking Batch status: {batch_id}...")
    
    batch = client.messages.batches.retrieve(batch_id)
    status = batch.processing_status
    counts = batch.request_counts

    print(f"   Current state: {status.upper()}")
    print(f"   Processing: {counts.processing}")
    print(f"   Successfully completed: {counts.succeeded}")
    print(f"   Errors: {counts.errored}")
    print(f"   Deleted: {counts.canceled}")

    if status != "ended":
        print("\nBatch not completed yet.")
        print("Check again later.")
        return

    print("\nBatch completed! Downloading results...")
    all_scored_results = []
    
    raw_logs = [] 

    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        
        log_entry = {
            "custom_id": custom_id,
            "status": result.result.type,
            "raw_text_received": None,
            "error_details": None
        }

        if result.result.type == "succeeded":
            raw_text = result.result.message.content[0].text.strip()
            log_entry["raw_text_received"] = raw_text

            # Rermoves markdown blocks (es. ```json ... ```) if Claude returns the data in that format
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[1]
                if "```" in raw_text:
                    raw_text = raw_text.rsplit("```", 1)[0].strip()

            try:
                # Converting the raw text to a list of dictionaries
                chunk_results = json.loads(raw_text)
                all_scored_results.extend(chunk_results)
                print(f"   {custom_id}: analyzing {len(chunk_results)} reviews")
            except json.JSONDecodeError as e:
                print(f"   {custom_id}: Decoding error JSON — {e}")
                log_entry["error_details"] = f"JSONDecodeError: {str(e)}"

        elif result.result.type == "errored":
            err = result.result.error
            print(f"   ✘ {custom_id}: Error API — {err.type}: {err.message}")
            log_entry["error_details"] = f"{err.type}: {err.message}"
            
        # Add the log entry for this result to the raw logs list
        raw_logs.append(log_entry)

    # Saving everything
    output_dir = "data/03_scored"
    os.makedirs(output_dir, exist_ok=True)

    # Saving raw logs for debugging and traceability
    if raw_logs:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{output_dir}/raw_responses_log_{batch_id}_{timestamp}.jsonl"
        
        with open(log_filename, "w", encoding="utf-8") as log_file:
            for log in raw_logs:
                log_file.write(json.dumps(log, ensure_ascii=False) + "\n")
        print(f"\nLog of raw reviews saved in: {log_filename}")

    # Saving excel
    if all_scored_results:
        print(f"Saving {len(all_scored_results)} reviews in Excel...")
        results_df = pd.DataFrame(all_scored_results)
        
        output_file = f"{output_dir}/data_ai_scored.xlsx"
        results_df.to_excel(output_file, index=False, engine="openpyxl")
        print(f"done! file saved in: {output_file}")
    else:
        print("\nNo result extracted.")

if __name__ == "__main__":
    check_and_download()
