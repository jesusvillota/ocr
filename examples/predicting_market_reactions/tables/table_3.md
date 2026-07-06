| Function             |  | Prompt                                                                 | Options                     |
|----------------------|--|------------------------------------------------------------------------|-----------------------------|
| 1. firms             |  | "List all the firms affected by the events<br>narrated in the article" | array                       |
| 1.1. firm            |  | "Iterate over each firm in firms"                                      | string                      |
| 1.2. ticker          |  | "State the stock market ticker of firm "                               | string                      |
|                      |  | "What type of shock does this article imply                            | {demand, supply, financial, |
| 1.3. shock_type      |  | on firm ?"                                                             | technology, policy}         |
|                      |  | "How much impact is this shock expected to                             | {minor, major}              |
| 1.4. shock_magnitude |  | have on firm?"                                                         |                             |
|                      |  | "In what direction is this shock expected to                           | {positive, negative}        |
| 1.5. shock_direction |  | impact firm?"                                                          |                             |
