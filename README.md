# A/B Testing for CatBoost Recommendation Service

## Overview

This repository contains the implementation of an **A/B testing framework** for a recommendation service using **CatBoost models**. The goal is to compare the performance of two different models by **randomly assigning users to one of two experimental groups** and tracking their responses.

## Purpose

The primary objective of this project is to:
1. **Evaluate the effectiveness of a new recommendation model** (test group) compared to the previous baseline model (control group).
2. **Implement an A/B testing framework** that splits users into two groups based on a hashing function.
3. **Log which model was applied to each user** to analyze performance metrics later.

## Implementation Details

### 1. **A/B Experimentation Setup**
- Users are randomly assigned to one of two groups using an **MD5 hash function** with a salt.
- **Control Group (`model_control`)**: Uses the original CatBoost model with standard hyperparameters.
- **Test Group (`model_test`)**: Uses an improved CatBoost model with tweaked hyperparameters.

### 2. **Recommendation Models**
The two models used in this project are:
- **Baseline Model (`model_control`)**:
  - `iterations=100`
  - `depth=6`
  - `learning_rate=0.1`
  - `l2_leaf_reg=3`
  
- **Test Model (`model_test`)**:
  - `iterations=200`
  - `depth=8`
  - `learning_rate=0.05`
  - `l2_leaf_reg=10`
  
Both models are trained on the same dataset with categorical features like `gender`, `country`, `city`, `topic`, etc.

### 3. **Service Structure**
- The service is built using **FastAPI**.
- **Users are assigned dynamically** to either the `control` or `test` group upon request.
- The response includes:
  - The **experiment group** ("control" or "test").
  - A list of **recommended posts** based on the assigned model.

### 4. **Data Handling**
- Data is stored in a **PostgreSQL database**.
- Features are loaded using **SQLAlchemy**.
- The `.env` file is used to securely store database credentials.

## API Usage

### **Endpoint: Get Post Recommendations**
```http
GET /post/recommendations/?id=<user_id>&time=<timestamp>&limit=<number>
