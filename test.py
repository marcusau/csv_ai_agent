import os
import pandas as pd
import chardet
from crewai import Crew, Task, Agent, LLM, Process
from crewai_tools import FileReadTool,  CSVSearchTool
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

current_dir = Path.cwd()
#print(f"Current working directory: {current_dir}")


llm = LLM(
    model='ollama/llama3.2:3b',
    base_url='http://localhost:11434',
)
csv_path=current_dir /'support_tickets_data.csv'

csv_tool = FileReadTool(file_path=csv_path)


dataset_inference_agent = Agent(
    role="Dataset Context Specialist",
    goal=(
        "Infer the context and purpose of the dataset by analyzing column names, data types, "
        "and a few sample rows. Extract insights about the domain and the type of data provided."
    ),
    backstory=(
        "An expert in understanding datasets and identifying their purpose. You have a deep understanding of data science, "
        "machine learning, and data analysis."
    ),
    tools=[csv_tool],
    llm=llm,
    verbose=True,
    allow_code_execution=False  # Changed to False since Docker is not running
)



dataset_inference_task = Task(
    description="""
    Analyze the dataset to determine its context, purpose, and structure. 
    This includes:
    - Examining Column Names: Identify meaningful column names and categorize them 
      based on their potential data types (numerical, categorical, date-time, etc.).
    - Sampling Data: Analyze a subset of rows to identify patterns, data relationships, 
      and potential anomalies in the dataset.
    - Domain and Application: Infer the dataset's domain, such as customer reviews, 
      sales data, or operational metrics, and suggest potential real-world applications.

    The goal is to provide stakeholders with an intuitive understanding of the dataset 
    and its potential uses without requiring them to delve into the raw data.
    """,
    expected_output="""
    A descriptive overview of the dataset's structure and purpose, highlighting:
    - Data columns and their inferred roles.
    - High-level insights into the type of data (e.g., transactional, temporal, or categorical).
    - Recommendations for possible applications or use cases.
    """,
    agent=dataset_inference_agent
)
########################################################

data_analysis_agent = Agent(
    role="Data Cleaning Specialist",
    goal=(
        "Analyze the dataset to identify missing values, incorrect data types, and potential outliers. "
        "Generate statistical summaries like mean, median, and correlations between variables."
    ),
    backstory=(
        "Specializes in cleaning and preparing data for analysis with expertise in data cleaning and preprocessing."
    ),
    tools=[csv_tool],
    llm=llm,
    verbose=True
)

data_analysis_task = Task(
    description="""
    Perform a comprehensive analysis of the dataset to identify missing values, incorrect data types, and potential outliers. 
    This includes:
    - Missing Values: Identify and quantify missing data across all columns, suggesting imputation or removal strategies.
    - Data Types: Validate and standardize column data types.
    - Statistical Summaries: Generate descriptive statistics like mean, median, and correlations for numerical columns.
    - Sampling Data: Analyze a subset of rows to identify patterns, data relationships, 
      and potential anomalies in the dataset.
    - Domain and Application: Infer the dataset's domain, such as customer reviews, 
      sales data, or operational metrics, and suggest potential real-world applications.

    The goal is to provide stakeholders with actionable insights and ensure the dataset is clean and ready for analysis.
    """,
    expected_output="""
    - A table or list of missing values and strategies for handling them.
    - Summary of identified data types and their standardization.
    - Statistical summaries for key columns, including correlations between variables.
    - Recommendations for further analysis or preprocessing.
    """,
    agent=data_analysis_agent
)
###############################################################
visualization_agent = Agent(
    role="Visualization Expert",
    goal=(
        "Generate meaningful visualizations such as histograms, scatter plots, line plots, bar charts, "
        "and heatmaps to provide insights into the data. Save all visualizations to a 'graphs/' directory."
    ),
    backstory=(
        "Specializes in creating compelling and informative visualizations. You are an expert in Python, pandas, "
        "matplotlib, seaborn, and data visualization, capable of creating impactful data stories."
    ),
    tools=[csv_tool],
    llm=llm,
    verbose=True,
    allow_code_execution=False  # Changed to False since Docker is not running
)


visualization_task = Task(
    description="""
    Create meaningful visualizations dynamically based on the dataset's content. 
    The visualizations should include:
    - Histograms: For numerical columns to showcase data distributions.
    - Bar Charts: For categorical columns with limited unique values to highlight frequencies.
    - Correlation Heatmaps: To represent the correlations between numerical variables.
    - Scatter Plots and Line Plots: To show relationships and trends.
    - Any other relevant visualizations based on the dataset's content.

    Save the visualizations as image files in the 'graphs/' directory and ensure they are 
    properly labeled with titles, axis names, and legends.

    These visualizations aim to uncover key patterns and relationships in the dataset.
    """,
    expected_output="""
    - A set of graphs saved in the 'graphs/' directory.
    - Each graph is annotated and labeled for clarity and ready for embedding in the final report.
    use the matplotlib library to create graphs.

    """,
    agent=visualization_agent
)
########################################################

markdown_report_agent = Agent(
    role="Report Specialist",
    goal=(
        "Compile all findings, analysis, and visualizations into a structured markdown report. "
        "Embed graphs and provide clear sections for analysis and summary."
    ),
    backstory="An expert in synthesizing data insights into polished reports.",
    tools=[csv_tool],
    llm=llm,
    verbose=True,
)

markdown_report_task = Task(
    description="""
    Create a detailed markdown report summarizing all analysis and visualizations. 
    The report should include:
    - Dataset Overview: Key insights from the context analysis, such as dataset 
      structure and inferred purpose.
    - Data Cleaning Summary: Detailed information on missing data, outliers, 
      and cleaning steps performed.
    - Statistical Summary: Tables and summaries of computed descriptive statistics.
    - Visualizations: Embedded images of charts created in the visualization task, 
      with captions explaining the insights.
    - Recommendations: Suggestions for further data preprocessing, modeling, 
      or potential use cases for the dataset.

    The report should be:
    - Organized into clearly labeled sections with a logical flow.
    - Formatted for stakeholders who may not have technical expertise.
    - Focused on actionable insights and takeaways.
    """,
    expected_output="""
    - A markdown report saved as 'report.md'.
    - Includes sections for analysis, summaries, and embedded visualizations.
    The Graphs and with bar and line plots and so on 
    - Provides actionable insights and recommendations.
    add the graphs in place of wherte it is necessary make sure you give it as detailed and well formatted as possible
    """,
    agent=markdown_report_agent,
    context=[dataset_inference_task, data_analysis_task, visualization_task],
    output_file='report.md'
)

csv_analysis_crew = Crew(
    agents=[
        dataset_inference_agent,
        data_analysis_agent,
        visualization_agent,
        markdown_report_agent
    ],
    tasks=[dataset_inference_task, data_analysis_task, visualization_task, markdown_report_task],
    process=Process.sequential,
    verbose=True
)

result = csv_analysis_crew.kickoff()
print("Crew Execution Complete. Final report generated.")
print(result)