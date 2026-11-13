import csv
import time
from contextlib import redirect_stdout
from pipeline import run_research_pipeline

# Read topics
with open("test_topics.txt", "r", encoding="utf-8") as file:
    topics = [line.strip() for line in file if line.strip()]

results = []

# First test only one topic
TEST_LIMIT = 1

for index, topic in enumerate(topics[:TEST_LIMIT], start=1):

    print(f"\nTesting {index}/{TEST_LIMIT}: {topic}")

    start_time = time.time()

    try:
        # Save detailed pipeline output separately
        with open(f"multiagent_log_{index}.txt", "w", encoding="utf-8") as log:
            with redirect_stdout(log):
                state = run_research_pipeline(topic)

        elapsed = round(time.time() - start_time, 2)

        results.append({
            "Topic": topic,
            "Execution_Time_Seconds": elapsed,
            "Report": state.get("report", ""),
            "Verification": state.get("verification", ""),
            "Critic_Feedback": state.get("feedback", ""),
            "Status": "Success"
        })

        print(f"Completed in {elapsed} seconds")

    except Exception as e:

        elapsed = round(time.time() - start_time, 2)

        results.append({
            "Topic": topic,
            "Execution_Time_Seconds": elapsed,
            "Report": "",
            "Verification": "",
            "Critic_Feedback": "",
            "Status": f"Failed: {str(e)}"
        })

        print(f"Failed: {e}")

# Save results
with open("multiagent_results.csv", "w", newline="", encoding="utf-8-sig") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Topic",
            "Execution_Time_Seconds",
            "Report",
            "Verification",
            "Critic_Feedback",
            "Status"
        ]
    )

    writer.writeheader()
    writer.writerows(results)

print("\nMulti-Agent testing completed.")
print("Results saved in multiagent_results.csv")