import pandas as pd
def read_parquet(file):
    df = pd.read_parquet(file)
    return df

def save_parquet(df, file):
    df.to_parquet(file, index=True)

input_file = "data/dapo-math-17k.parquet"
output_file = "data/dapo-math-17k.parquet"

dapo_df = read_parquet(input_file)
# Create a function to modify the prompt content
def modify_prompt(prompt_list):
    for i in range(len(prompt_list)):
        prompt_list[i]['content'] = prompt_list[i]['content'].replace("\n\n", " ").replace(
            "Remember to put your answer on its own line after \"Answer:\".",
            "Please output the final answer within \\boxed{}."
        ).replace(
            "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem. ",
            ""
        )
    return prompt_list

# Apply the function to modify prompts
dapo_df['prompt'] = dapo_df['prompt'].apply(modify_prompt)

# Save the modified dataframe
save_parquet(dapo_df, output_file)