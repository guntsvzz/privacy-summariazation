# Privacy Summarization

## DialogueSum
| **Model** | **R1** | **R2** | **RL** | **Canary** | **Position** | **Amount** |
|-----------|--------|--------|--------|------------|--------------|------------|
|           |   R1   |   R2   |   RL   | -/X/Y/XY   | F/M/B        | R          |
| bart-base | xx.xx  | xx.xx  | xx.xx  | -          | B            | 1000       |
| bart-base | xx.xx  | xx.xx  | xx.xx  | X          | B            | 1000       |
| bart-base | xx.xx  | xx.xx  | xx.xx  | Y          | B            | 1000       |
| bart-base | xx.xx  | xx.xx  | xx.xx  | XY         | B            | 1000       |

## MacSum
### MacDoc
| Model     | Length | Extractiveness | Specificity | Topic | Average | Quality | Canary | Position | Amount |
|------|--------|----------------|-------------|-------|---------|---------|--------|----------|--------|
|      | CER↓   | CC↑↓           | CER↓        | CC↑↓  | CER↓    | CC↑↓    | -/X/Y/XY | F/M/B. | R      |
| bart-base | xx.xx  | xx.xx          | xx.xx       | xx.xx | xx.xx   | xx.xx   | xx.xx  | xx.xx    | -  | xx.xx  | xx.xx  |
| bart-base | xx.xx  | xx.xx          | xx.xx       | xx.xx | xx.xx   | xx.xx   | xx.xx  | xx.xx    | X  | xx.xx  | xx.xx  |
| bart-base | xx.xx  | xx.xx          | xx.xx       | xx.xx | xx.xx   | xx.xx   | xx.xx  | xx.xx    | Y  | xx.xx  | xx.xx  |
| bart-base | xx.xx  | xx.xx          | xx.xx       | xx.xx | xx.xx   | xx.xx   | xx.xx  | xx.xx    | XY  | xx.xx  | xx.xx  |


### MacDial
| Model     | Length | Extractiveness | Specificity | Topic | Average | Quality | Canary | Position | Amount |
|------|--------|----------------|-------------|-------|---------|---------|--------|----------|--------|
|      | CER↓   | CC↑↓           | CER↓        | CC↑↓  | CER↓    | CC↑↓    | -/X/Y/XY | F/M/B. | R      |
| bart-base | xx.xx  | xx.xx          | xx.xx       | xx.xx | xx.xx   | xx.xx    | -  | xx.xx  | xx.xx  |
| bart-base | xx.xx  | xx.xx          | xx.xx       | xx.xx | xx.xx   | xx.xx    | X  | xx.xx  | xx.xx  |
| bart-base | xx.xx  | xx.xx          | xx.xx       | xx.xx | xx.xx   | xx.xx    | Y  | xx.xx  | xx.xx  |
| bart-base | xx.xx  | xx.xx          | xx.xx       | xx.xx | xx.xx   | xx.xx    | XY  | xx.xx  | xx.xx  |
