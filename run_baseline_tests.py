import csv
from baseline import run_baseline

# Read research topics
with open("test_topics.txt", "r", encoding="utf-8") as file:
    topics = [line.strip() for line in file if line.strip()]

results = []

for index, topic in enumerate(topics, start=1):
    print(f"\nTesting {index}/{len(topics)}: {topic}")

    try:
        report, elapsed = run_baseline(topic)

        results.append({
            "Topic": topic,
            "Execution_Time_Seconds": elapsed,
            "Report": report,
            "Status": "Success"
        })

        print(f"Completed in {elapsed} seconds")

    except Exception as e:
        results.append({
            "Topic": topic,
            "Execution_Time_Seconds": "",
            "Report": "",
            "Status": f"Failed: {str(e)}"
        })

# Save results
with open("baseline_results.csv", "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Topic",
            "Execution_Time_Seconds",
            "Report",
            "Status"
        ]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nBaseline testing completed.")
print("Results saved in baseline_results.csv")