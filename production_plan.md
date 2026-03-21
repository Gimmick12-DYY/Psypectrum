# Production Plan: Emotion-Color Augmented GNN-BERT
**Project Goal:** Enhance Sentiment Analysis on the GoEmotions dataset by integrating a confidence-weighted, hue-based domain knowledge feature into a Graph Neural Network.

---

## 1. Feature Engineering Logic
[cite_start]The model utilizes a deterministic mapping of emotions to a 3D polar coordinate system (Hue, Saturation, Valence) to provide structural "priors" to the GNN[cite: 1, 2].

### 1.1 The Color Map (Core Constants)
* [cite_start]**Positive Anchor (Warm):** `40°` (Orange-ish)[cite: 3].
* [cite_start]**Negative Anchor (Cold):** `220°` (Blue-ish)[cite: 3].
* [cite_start]**Neutral/Surprise Anchor:** `130°` (Green/Teal)[cite: 38, 39, 40].

### 1.2 Mathematical Transformation
[cite_start]To avoid wrap-around discontinuities at 0°/360°, all hues are converted into unit vectors (sin, cos)[cite: 42]:
* [cite_start]**Hue Formula:** $h = 40 + (220 - 40) \times \frac{1 - v}{2}$[cite: 3].
* [cite_start]**Coordinate Vector:** $E_{vector} = \text{Confidence} \cdot \begin{bmatrix} \text{Saturation} \cdot \cos(h) \\ \text{Saturation} \cdot \sin(h) \\ \text{Valence} \end{bmatrix}$[cite: 42].

| Variable | Definition | Logic |
| :--- | :--- | :--- |
| **Valence ($v$)** | Emotional polarity | [cite_start]Provided by `COLOR_MAP.txt` [-1 to 1][cite: 2, 4]. |
| **Saturation ($s$)** | Emotional intensity | 0.0 for Neutral, 1.0 for Strong Emotions. |
| **Confidence** | Model certainty | Softmax output from the BERT backbone. |

---

## 2. System Architecture
The model consists of three primary stages:

### Phase A: The Semantic Backbone (BERT)
* [cite_start]**Model:** `bert-base-uncased` fine-tuned on GoEmotions[cite: 1].
* **Task:** Generate a 768-dimensional embedding for each text node.
* **Output:** Hidden state $H_{bert}$ and the probability distribution across 28 classes.

### Phase B: The Color Projection (The Bridge)
* [cite_start]**Input:** The 3D $E_{vector}$ (Cos, Sin, Valence)[cite: 42].
* **Transformation:** A Linear layer $W_c \in \mathbb{R}^{3 \times 128}$ to project the small color vector into a high-dimensional space compatible with BERT.
* **Activation:** ReLU or GeLU to introduce non-linearity.

### Phase C: Graph Reasoning (GNN)
* **Graph Construction:** Nodes represent sentences/documents. Edges are weighted by word co-occurrence or cosine similarity.
* **Node Initialization:** $X_i = [H_{bert} \, \Vert \, \text{Projected\_Color}_i]$ (Concatenation).
* **Message Passing:** A Graph Convolutional Network (GCN) or Graph Attention Network (GAT) aggregates emotional neighbors.

---

## 3. Implementation Checklist for Agents

1. [cite_start]**[ ] Data Preparation:** Parse `COLOR_MAP.txt` into a lookup table for all 28 GoEmotions labels[cite: 4].
2. [cite_start]**[ ] Neutral Safeguard:** Implement a hard-coded saturation of $0$ for the `neutral` label to ensure it sits at the origin of the emotion wheel[cite: 39].
3. **[ ] Feature Scaling:** Ensure the BERT 768D output and the Projected Color 128D output are Layer-Normalized before concatenation.
4. **[ ] Training Strategy:** * **Warm-up:** Train the BERT classifier first to stabilize Confidence scores.
    * **Joint Training:** Fine-tune the GNN and the Projection layer together while keeping BERT weights frozen (initially).
5. **[ ] Evaluation:** Compare results against a baseline "BERT-only" model to measure the accuracy gain provided by the Color-GNN logic.

---

## 4. Potential Risks & Mitigations
* **Redundant Logic:** If BERT and Color provide identical information, add a **Residual Connection** from BERT directly to the final classifier.
* [cite_start]**Label Collisions:** For `Surprise` and `Neutral` (both at 130°), use the **Valence** dimension as the tie-breaker to distinguish between the two[cite: 38, 39, 40].