# Clinical NLP Pipeline for Medical Triage

## Overview
Hospitals process thousands of "Discharge Summaries" and "Clinical Notes" daily, requiring doctors to spend hours manually reading and categorizing patients by medical specialty. 

This project is an end-to-end Natural Language Processing (NLP) pipeline that automates this triage process. It cleans raw, messy medical text, extracts critical medical entities (symptoms and diseases), detects negated symptoms (e.g., distinguishing "chest pain" from "denies chest pain"), and automatically classifies the text into the correct medical department.

## Technical Stack
* **Language:** Python
* **NLP Libraries:** spaCy, NLTK
* **Machine Learning:** Scikit-Learn (TF-IDF Vectorization, [Insert Classifier you used, e.g., Random Forest/LinearSVC])
* **Data Processing:** Pandas, Regular Expressions (Regex)

## Key Features & Implementation
1. **Clinical Preprocessing:** Used Regex to strip header noise and engineered a custom dictionary to map and expand common medical abbreviations (e.g., `pt` -> patient, `hx` -> history).
2. **Entity Extraction (POS Tagging):** Utilized NLP Part-of-Speech tagging to isolate nouns (symptoms) and adjectives (severity modifiers).
3. **Dependency Parsing & Negation Detection:** Leveraged `spaCy` dependency trees to analyze the syntactic relationship between words, successfully identifying when a symptom was denied by the patient rather than experienced.
4. **Specialty Classification:** Vectorized the cleaned transcriptions using TF-IDF and trained a machine learning classifier to route the reports to 4 primary departments (Surgery, Internal Medicine, Orthopedics, Cardiology).

## Results
* Achieved an overall classification accuracy of **[Insert your accuracy %, e.g., 85%]**.
* Successfully isolated department-specific vocabulary (e.g., Heart/Artery for Cardiology vs. Bone/Joint for Orthopedics).
* **[Optional: Insert any interesting observation from your confusion matrix, e.g., "The model occasionally confused Surgery with Orthopedics due to overlapping terminology regarding incisions and bone repair."]**

## Dataset
Trained on a subset of the [MTSamples Dataset](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions), filtered for the top most common medical specialties to ensure balanced class distributions.
