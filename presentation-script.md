# Presentation Script: Hand Gesture Recognition for Puzzle Game Control

**Estimated Time:** 10 Minutes (~30 seconds per slide)
**Pacing:** Keep a steady, conversational tone. Don’t rush the math/data slides; they are where the grade is earned.

---

### [Slide 1: Title Slide] (0:00 - 0:30)
**Speaker (Hiếu):** 
"Hello everyone, and Professor Dung. We are team members Hiếu, Nhất Anh, and Quốc Anh. Today, we’re thrilled to present our Computer Vision project: 'Hand Gesture Recognition for Interactive Puzzle Game Control'. We’ve built a system that lets you play a jigsaw puzzle in your browser using only a webcam, and we will be comparing three entirely different approaches to making this work."

---

### [Slide 2: Motivation] (0:30 - 1:00)
**Speaker:**
"To start with the 'Why'—traditional inputs like a mouse break our immersion. If you want to interact with a screen naturally, hands are the best biological tool we have. But engineering a touchless system is hard. Hand sizes vary, skin tones differ, and room lighting changes constantly. Our core challenge was: How do we build a pipeline that is blind to these physical differences and focuses purely on structural intent, without relying on excessively heavy deep learning?"

---

### [Slide 3: Objectives] (1:00 - 1:30)
**Speaker:**
"We set out with three main objectives. First, we implemented and compared three methods spanning the complexity spectrum: a Rule-based Heuristic, a Random Forest with engineered features, and a Multi-Layer Perceptron neural network. Second, we rigorously evaluated them using 'person-aware cross-validation' on a massive dataset. Finally, we proved our best model’s capability by deploying it in a zero-latency browser game."

---

### [Slide 4: Gesture Vocabulary] (1:30 - 1:50)
**Speaker:**
"For our game, we defined a vocabulary of five key states: 
- 'Open Hand' to release a piece
- 'Fist' to rapidly rotate it
- 'Pinch' to grab it
- A two-handed 'Frame' gesture
- And a 'None' state where the hand is just resting. Every gesture maps instantly to the canvas physics."

---

### [Slide 5: Pipeline Architecture] (1:50 - 2:15)
**Speaker:**
"Here is the end-to-end flow. The webcam grabs a video frame and feeds it to Google’s MediaPipe, which extracts exactly 21 joint coordinates. We then run a rigorous mathematical normalization step to achieve scale and translation invariance. Finally, those normalized features hit our classifier—whether that’s our Random Forest or our Neural Network—to predict the user's intent."

---

### [Slide 6: Training Data (HaGRID v2)] (2:15 - 2:45)
**Speaker:**
"To train our models, we used the HaGRID v2 dataset, drawing exactly 9,231 samples from over 3,600 unique people. It’s important to note a crucial limitation here: HaGRID provides highly accurate X and Y coordinates, but the Z-depth is set to zero. As we’ll see in a moment, treating a flat 2D projection as 3D geometry severely penalizes any traditional, math-heavy heuristic."

---

### [Slide 7: Feature Extraction] (2:45 - 3:20)
**Speaker:**
"This slide is the mathematical core of our pipeline. Raw coordinates are useless if a user simply steps backward from the camera. 
We fix this in two steps. First, Translation Invariance: we subtract the wrist coordinate from all other joints, anchoring the hand to the origin $(0,0,0)$. Second, Scale Invariance: we divide every joint by the Euclidean distance to the middle finger. This standardizes a child's hand to be mathematically identical to an adult's hand, yielding a pure 60-dimensional structural vector."

---

### [Slide 8: Method 1 - Heuristic] (3:20 - 3:45)
**Speaker:**
"Our first approach was a Rule-Based Heuristic using 20 manual angle and distance thresholds. 
As expected due to the dataset limitation I mentioned earlier, it failed completely, achieving just a 0.99% accuracy rate. Because we lacked the Z-depth geometry, the angle calculations collapsed into noise. This failure perfectly justified our need to switch to data-driven machine learning models to find the hidden patterns."

---

### [Slide 9: Method 2 - Random Forest] (3:45 - 4:15)
**Speaker:**
"Our second approach was a Random Forest using 100 decision trees. Because decision trees struggle with raw 60-dimensional coordinates, we engineered a specific 24-dimensional feature vector of just the pairwise distances between fingertips. 
It worked exceptionally well, achieving 94.13% accuracy with a blazing fast 13-millisecond latency. Best of all, as you can see in the graph, it is highly interpretable—we can see exactly which distances the trees prioritized."

---

### [Slide 10: Method 3 - MLP] (4:15 - 4:45)
**Speaker:**
"Our final approach was Deep Learning: a Multi-Layer Perceptron (MLP). We used a lightweight two-hidden-layer network with dropout. 
Unlike the Random Forest, the MLP didn't need our feature engineering; we fed it the 60 raw coordinates directly. It achieved our peak accuracy of 98.41%. Its footprint is tiny—only 227 kilobytes—making it incredibly efficient."

---

### [Slide 11: Summary Table] (4:45 - 5:15)
**Speaker:**
"Here is the head-to-head comparison. The Heuristic fails but uses zero memory. The Random Forest is highly accurate and very fast at 13 milliseconds, but requires 8.6 MB of space. The MLP dominates in accuracy (98.4%) and is the smallest ML model at just 227 KB, though it is slightly slower at 43 milliseconds. We had to decide which tradeoff mattered most."

---

### [Slide 12: Person-Aware CV] (5:15 - 5:45)
**Speaker:**
"Before checking the errors, we need to address data leakage. We didn't just randomly split our dataset. We used 'Group 5-Fold Cross-Validation' grouped by the Person ID. This guarantees that a subject used to train the model is never seen during the test phase. Our 98% accuracy is a true reflection of the model generalizing to completely unseen humans, not just memorizing the hands it already knows."

---

### [Slide 13: Accuracy vs Latency] (5:45 - 6:30)
**Speaker:**
"The ultimate decision came down to hardware latency constraints. To hit a smooth 60 frames per second on screen, inference must happen in under 16.7 milliseconds. Our Random Forest easily beats this at 13.2 ms. The MLP takes 43.7 ms, giving us about 23 real-world FPS. 
However, human gestures transition at about 1 to 2 times a second. Because 23 FPS is more than enough temporal resolution to capture human intent, we chose to deploy the MLP to maximize accuracy over unnecessary rendering speed."

---

### [Slide 14: Error Analysis] (6:30 - 7:00)
**Speaker:**
"When we look at where the models fail, the Random Forest heavily struggles with the 'Frame' gesture, scoring an F1 of 89%. This is because it drops inter-hand context when trees make isolated binary splits. The MLP, through its dense, fully connected layers, perfectly captures the relationships between the two hands, boosting the Frame accuracy by over 8 full percentage points up to 97.3%."

---

### [Slide 15: Per-Class F1 Scores] (7:00 - 7:30)
**Speaker:**
"This graph clearly visualizes that MLP dominance across every single class. The most critical gains were in complex structural gestures like the Pinch and the Frame, where the neural network's capacity for generalized representation far outweighed the manual feature engineering of the Random Forest."

---

### [Slide 16: CV Stability] (7:30 - 8:00)
**Speaker:**
"Looking across all 5 folds, the MLP is incredibly stable with a standard deviation of only 0.26%. It is completely invariant to the test population. The Random Forest, on the other hand, dips in fold 4, suggesting it occasionally overfits to the geometric variations of specific subjects."

---

### [Slide 17: Demonstration] (8:00 - 8:45)
*(At this point, switch to the browser to show the game or play the video if pre-recorded)*

**Speaker:**
"We will now demonstrate the system live. The entire pipeline—from MediaPipe landmark detection to the MLP ONNX Runtime classification—is happening entirely client-side in the browser. There is no backend server running anywhere. As you can see, the Pinch seamlessly grabs the piece, and the Fist rotates it, controlled entirely by the gesture intent we mapped."

---

### [Slide 18: Software Engineering] (8:45 - 9:15)
**Speaker:**
"Behind the scenes, the repository is built with production standards. We have 28 automated test files covering the entire preprocessing and feature extraction logic, ensuring our mathematical transformations never regress. The code is modular, separating the data pipelines from the PyTorch evaluation logic cleanly."

---

### [Slide 19: Limitations & Future] (9:15 - 9:40)
**Speaker:**
"Regarding limitations, our system currently treats every frame as an isolated event. If there is a one-frame glitch, the prediction stutters. While we used a simple JavaScript debounce buffer to smooth the gameplay, a future extension would involve true temporal modeling—feeding a sliding window of the last 10 frames into a 1D-CNN or an LSTM to interpret motion over time rather than static photographs."

---

### [Slide 20: Conclusion] (9:40 - 10:00)
**Speaker:**
"To summarize: Machine learning outclasses geometric heuristics by a stunning margin when data is limited to 2D. We proved that an MLP achieves optimal 98.4% accuracy with zero feature engineering and fits into a tiny 227KB footprint perfectly suited for browser deployment. 

Thank you for your time and attention. We are happy to take any questions."
