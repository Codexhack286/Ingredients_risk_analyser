
# Ingredient Risk Classification Dataset Documentation

## Overview
This synthetic dataset is designed for training NLP models (particularly transformers like BERT/DeBERTa) 
to classify food ingredient lists into 5 risk levels. The dataset was generated from the 
"FOOD PRODUCTS INGREDIENT DATASET" (Mendeley Data, DOI: 10.17632/58mfpfxksk.1).

## Source Dataset
- **Title**: Food Products Ingredient Dataset
- **Published**: January 22, 2024
- **Contributors**: Rajesh Kumar, Dilbag Singh (Chaudhary Devi Lal University)
- **License**: CC BY 4.0
- **Original Purpose**: ML-based food product classification through ingredients
- **Contains**: 324 ingredients from 7 product categories with Natural/Artificial and Processed/Unprocessed labels

## Risk Level Classification Schema

### Level 1: VERY SAFE
- **Description**: Natural, unprocessed, whole foods
- **Examples**: Water, milk solids, whole wheat, fresh vegetables, spices, nuts, seeds, pulses
- **Criteria**: Natural (0) AND Unprocessed (0), no additive codes

### Level 2: SAFE  
- **Description**: Natural ingredients, minimally processed
- **Examples**: Sugar, butter, flour, iodized salt, vegetable oils, malt extract
- **Criteria**: Natural (0) with some processing, common kitchen ingredients

### Level 3: MODERATE
- **Description**: Refined/processed ingredients, mild additives
- **Examples**: Refined wheat flour (maida), palm oil, refined sugar, liquid glucose, antioxidants (INS 300)
- **Criteria**: Refined ingredients or mild processing aids

### Level 4: CONCERNING
- **Description**: Artificial additives, preservatives, emulsifiers
- **Examples**: Preservatives (282, 211), emulsifiers (471, 481), stabilizers, acidity regulators, colors (INS 150)
- **Criteria**: Artificial (1) AND Processed (1), additive codes present

### Level 5: HIGH RISK
- **Description**: Synthetic colors, artificial sweeteners, harmful additives
- **Examples**: Synthetic food colors (INS 102, 110, 122, 124, 133), artificial sweeteners (951, 950, 955), 
  sodium benzoate (INS 211), BHA/BHT (INS 320/321), flavor enhancers (INS 627, 631, 635)
- **Criteria**: Known harmful additives per CSPI ratings, synthetic dyes, artificial sweeteners

## Labeling Methodology

The risk scoring is based on:

1. **CSPI Chemical Cuisine Ratings**: The Center for Science in the Public Interest rates food additives 
   from "Safe" to "Avoid" based on scientific evidence review.

2. **NOVA Food Classification**: Groups foods by degree of processing (unprocessed → ultra-processed).

3. **INS/E-Number Categorization**: International Numbering System codes indicate additive types:
   - E100-199: Colors
   - E200-299: Preservatives  
   - E300-399: Antioxidants, acidity regulators
   - E400-499: Thickeners, stabilizers, emulsifiers
   - E500-599: Acidity regulators, anti-caking agents
   - E600-699: Flavor enhancers
   - E900-999: Sweeteners, glazing agents

4. **Research-backed risk factors**:
   - Synthetic food dyes linked to behavioral issues in children
   - Artificial sweeteners classified as possibly carcinogenic (IARC)
   - Sodium nitrite/nitrate concerns in processed meats
   - Trans fats from hydrogenated oils
   - BHA/BHT antioxidant concerns

## Dataset Statistics

- **Total Samples**: 2,424
- **Single Ingredient Samples**: 324 (from original dataset)
- **Multi-Ingredient Samples**: 1,200 (3-12 ingredients combined)
- **Balanced Samples**: 900 (class-balancing additions)

### Class Distribution:
| Risk Level | Label       | Count |
|------------|-------------|-------|
| 1          | Very Safe   | 400   |
| 2          | Safe        | 400   |
| 3          | Moderate    | 400   |
| 4          | Concerning  | 599   |
| 5          | High Risk   | 625   |

### Text Statistics:
- Mean length: ~106 characters
- Max length: 359 characters
- Min length: 4 characters

## Multi-Ingredient Risk Assignment

For multi-ingredient samples, the **maximum risk level** among all ingredients is assigned 
(conservative approach). This mirrors real-world safety assessment where the presence of 
any high-risk ingredient elevates overall product concern.

## Usage for Transformer Training

```python
from datasets import Dataset
import pandas as pd

# Load dataset
df = pd.read_csv('ingredient_risk_classification_dataset.csv')

# Convert to HuggingFace dataset
dataset = Dataset.from_pandas(df)

# For classification, use 'risk_level' (1-5) or convert to 0-indexed:
df['label'] = df['risk_level'] - 1  # 0-4 for model training
```

## Recommended Model Architecture

- **Base Model**: DeBERTa-v3-base or BERT-base
- **Classification Head**: Single-label classification (5 classes)
- **Loss Function**: CrossEntropyLoss with class weights for imbalance
- **Max Sequence Length**: 512 tokens (sufficient for longest samples)

## References

1. Kumar, R. & Singh, D. (2024). Food Products Ingredient Dataset. Mendeley Data.
2. CSPI Chemical Cuisine Database. https://www.cspi.org/
3. Monteiro, C.A. et al. (2017). NOVA food classification. Public Health Nutrition.
4. Jiang, T. et al. (2022). Transformer-based food safety risk level prediction. Foods.
5. WHO/JECFA Food Additive Evaluations

## License
This synthetic dataset inherits CC BY 4.0 from the source dataset.
Attribution required: Kumar & Singh, Chaudhary Devi Lal University.
