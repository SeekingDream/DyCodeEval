

SCENARIO_PT = """
Suggest real-world scenarios that provide meaningful context in the following areas: {s1}, {s2}, {s3}, {s4}, {s5}, and any other practical fields. Each scenario should be general but applicable, providing useful insight for potential applications.

For clarity, return each scenario on a separate line without additional explanation. Use the example below for reference.

Please put your suggested Real-world Scenarios in <scenario></scenario> tags.

# Example:
<example>
{exp}
</example>
"""

CONTEXT_PT = """
I have a natural language problem description, input types, and a real-world scenario. For each variable in the problem, provide a meaningful context tailored to the given scenario. The context should explain how each variable is involved in or relates to the scenario, ensuring practical relevance.

Please ensure for to put your meaningful context in <context></context> tags.

# Problem Description:
<problem_description>
{pb}
</problem_description>

# Input Types:
<input_types>
{var}
</input_types>

# Real-world Scenario:
<scenario>
{scenario}
</scenario>

# Instructions:
- For each variable in the input types, generate only one context that highlights its role or significance within the problem and scenario.
- The context should help to clarify the variable’s meaning and importance, ensuring that it fits into the given real-world scenario.
- Provide only the contexts for the variables (no additional reasoning steps).

# Example:
For the Problem Description:  
<problem_description>
Determine if the average temperature in a city exceeds a certain threshold during a week.
</problem_description>

Input Types:  
<input_types>
temperatures: list of float  
threshold: float  
</input_types>  

Scenario:  
<scenario>
Climate Analysis - Monitoring Urban Heat Trends  
</scenario>  

Generated Contexts:  
<context>
temperatures: Daily recorded temperatures in a city, analyzed for urban heat trends.  
threshold: Critical temperature level indicating hazardous or abnormal heat. 
</context>  
"""

REWRITE_PT = """
Given a seed programming problem description, a selected real-world scenario, and contextualized input variables, rewrite the original problem to make it relevant to the scenario. The rewritten problem should:
- Preserve the original problem's complexity and constraints.
- Ensure the problem remains solvable with the same solution approach.
- Be clear, concise, and logically consistent within the new context.
- Just return the rewritten problem description without any additional commentary or steps, and do not include any input output demons in your problem description.
- Limit the new rewritten problem description to {num} - {num_2} sentences.
- Make sure your rewritten problem description is clear, concise and contains no unnecessary information.

Please ensure to put your rewritten problem description in <new_problem></new_problem> tags.

# Problem Description:
{pb}

# Real-World Scenario:
{scenario}

# Contextualized Input Variables:
{input_variables}

"""

VERIFY_PT = """
Assess whether the two given natural language instructions convey the same meaning. Respond with 'Yes' if they do, or 'No' if they do not.

Please ensure your answer is either "Yes" or "No".

# Instrcution A:
{inst_a}

# Instrcution B:
{inst_b}

# Your Answer:
"""