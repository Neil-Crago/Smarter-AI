# Smarter AI

[](https://www.python.org/)
[](https://opensource.org/licenses/MIT)
[](https://www.google.com/search?q=)

`Smarter AI` is a sophisticated, multi-faceted quality assessment engine for Large Language Model (LLM) responses. It moves beyond simple metrics to provide a nuanced, data-driven quality score by analyzing a response across multiple dimensions of linguistic and factual quality.

The system is designed as a modular "Quality Oracle" that can be integrated into any LLM-based application (like the `fa_slow_ai` project) to ensure the accuracy, coherence, and relevance of its outputs.

## Core Concept

Traditional AI evaluation often relies on a single metric or human oversight. `Smarter AI` operates on the principle that response quality is not a single property, but an **ensemble of signals**. By extracting a rich feature set from a given `(prompt, response)` pair and feeding it into a machine learning classifier, the system learns to identify the subtle patterns that distinguish high-quality responses from subtly flawed or nonsensical ones.

The project evolved from a simple signal processing concept (FFT on text) into a comprehensive, AI-powered diagnostic engine.

## Features

  * **Multi-Feature Analysis:** Evaluates responses across seven distinct, complementary dimensions.
  * **Hybrid Flow Analysis:**
      * **FFT Score:** Measures local, high-frequency semantic consistency.
      * **Lomb-Scargle Score:** Measures the overall, non-uniform "linguistic rhythm" of the response.
  * **Topic Coherence:** Detects internal topic drift and outlier sentences.
  * **Relevance Score:** Measures the semantic similarity of the response to the original prompt.
  * **Repetition Penalty:** Identifies and penalizes unnatural, redundant phrasing using n-gram analysis.
  * **AI as a Judge:** Leverages a powerful external LLM (e.g., Gemini) to score abstract qualities that are difficult to measure algorithmically:
      * **Factual Accuracy Score**
      * **Logical Reasoning Score**
  * **Machine Learning Classifier:** Uses a trained Logistic Regression model to intelligently weigh all seven features and produce a final, nuanced quality score (0.0 to 1.0).
  * **Efficient Caching:** Includes a mechanism to save and load the generated feature set, avoiding the need to re-run expensive API calls during development and testing.

## Architecture & Workflow

The system operates as a straightforward pipeline:

`User (Prompt, Response) -> Feature Extraction Engine -> Feature Vector [7D] -> Scaler -> ML Classifier -> Final Quality Score`

1.  **Input:** A `(prompt, response)` text pair.
2.  **Feature Extraction:** The seven feature functions are run to generate a feature vector.
3.  **Scaling:** The features are scaled to a standard range for optimal model performance.
4.  **Classification:** The trained Logistic Regression model predicts the probability that the response is "good."
5.  **Output:** A single, reliable quality score between 0.0 and 1.0.

## Installation

The project is built in Python and relies on several standard data science and AI libraries.

1.  **Install required libraries:**

    ```bash
    pip install numpy scikit-learn sentence-transformers scipy google-generativeai
    ```

2.  **Set up your API Key:**
    The "AI as a Judge" feature requires an API key for the Gemini API. Set this as an environment variable for security.

      * **Linux/macOS:** `export GEMINI_API_KEY="YOUR_API_KEY_HERE"`
      * **Windows:** `set GEMINI_API_KEY="YOUR_API_KEY_HERE"`

## Usage

Once the classifier has been trained (by running the main script on a labeled dataset), it can be saved and used to score new responses.

```python
import joblib # For saving/loading the trained model

# --- Assume 'classifier' and 'scaler' are your trained objects ---

# Save the trained model and scaler to disk
joblib.dump(classifier, 'smarter_ai_classifier.pkl')
joblib.dump(scaler, 'smarter_ai_scaler.pkl')

# --- In a different application, load the model to score a new response ---

loaded_classifier = joblib.load('smarter_ai_classifier.pkl')
loaded_scaler = joblib.load('smarter_ai_scaler.pkl')

def score_new_response(prompt, response):
    """
    Uses the trained Smarter AI model to score a new response.
    """
    # 1. Calculate the 7 features for the new response
    features = [
        calculate_fft_score(response),
        calculate_lombscargle_score(response),
        calculate_topic_coherence(response),
        calculate_relevance_score(prompt, response),
        calculate_repetition_penalty(response),
        get_ai_judge_score(prompt, response, "factual accuracy"),
        get_ai_judge_score(prompt, response, "logical reasoning")
    ]
    
    # 2. Reshape and scale the features
    feature_vector = np.array(features).reshape(1, -1)
    scaled_features = loaded_scaler.transform(feature_vector)
    
    # 3. Predict the probability of being a "good" response
    probability_good = loaded_classifier.predict_proba(scaled_features)[0][1]
    
    return probability_good

# Example
new_prompt = "What is the capital of Canada?"
new_response = "The capital of Canada is Ottawa, located in the province of Ontario."
final_score = score_new_response(new_prompt, new_response)

print(f"The quality score for the new response is: {final_score:.4f}")

```

## Future Development (Roadmap)

`Smarter AI` is a strong foundation that can be extended with even more sophisticated capabilities.

  * **"Gold-Standard Expert":** Implement the "A\* Path" or "Error Vector" analysis (`v_good - v_bad`) as a new feature to directly measure how far a bad response deviates from its ideal counterpart.
  * **Continuous Learning Feedback Loop:** Create a mechanism where high-scoring responses are added to a "gold standard" library, which is then used to periodically fine-tune both the generator (`fa_slow_ai`) and the classifier (`Smarter AI`).
  * **Advanced Feature Engineering:**
      * **Smart Repetition:** Upgrade the n-gram penalty with Part-of-Speech (POS) tagging to distinguish stylistic repetition from redundancy.
      * **Rhythm Score:** Implement the iambic rhythm analysis to measure the phonetic elegance and cadence of the response.
  * **Prompt Improvement Modules:** Develop the "Smart Wizard" and "Proactive Augmentation" systems to improve user prompts *before* they are sent to the generator.


## Author:

Neil Crago

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.