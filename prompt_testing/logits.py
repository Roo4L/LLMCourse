import openai
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import os
import numpy as np

@dataclass
class CompletionResult:
    response_text: str
    logprobs: List[Dict[str, Any]]
    system_fingerprint: str
    
class ChatGPTPlayground:
    def __init__(self, api_key: str = None):
        """Initialize the playground with OpenAI API key."""
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        self.client = openai.OpenAI(api_key=api_key)
        
    def generate_completions(
        self,
        prompt: str,
        n_completions: int = 3,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int = None,
        model: str = "gpt-4o",
        max_tokens: int = 150
    ) -> List[CompletionResult]:
        """Generate multiple completions with specified parameters."""
        results = []
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                max_tokens=max_tokens,
                logprobs=True,
                top_logprobs=5,
                n=n_completions
            )

            for choice in response.choices:
                completion = CompletionResult(
                    response_text=choice.message.content,
                    logprobs=choice.logprobs,
                    system_fingerprint=response.system_fingerprint
                )
                results.append(completion)
                
        except Exception as e:
            print(f"Error generating completion: {str(e)}")
                
        return results

    def save_results(self, results: List[CompletionResult], parameters: Dict[str, Any]):
        """Save results and parameters to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = {
            "parameters": parameters,
            "results": [
                {
                    "response_text": result.response_text,
                    "logprobs": result.logprobs,
                    "system_fingerprint": result.system_fingerprint
                }
                for result in results
            ]
        }
        
        filename = f"completion_results_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {filename}")

    def print_results(self, results: List[CompletionResult]):
        """Print results in a readable format with logprobs, probabilities, and tokens."""
        for i, result in enumerate(results, 1):
            print(f"\nCompletion {i}:")
            print("-" * 50)
            print(f"System Fingerprint: {result.system_fingerprint}")
            print("Response with logprobs and probabilities:")
            
            if result.logprobs:
                chunk_size = 10
                token_infos = result.logprobs.content
                
                for i in range(0, len(token_infos), chunk_size):
                    chunk = token_infos[i:i + chunk_size]
                    
                    # Print logprobs line
                    logprob_line = " ".join(f"{info.logprob:.2f}".ljust(8) for info in chunk)
                    print("LogProbs:", logprob_line)
                    
                    # Print probabilities line
                    prob_line = " ".join(f"{np.exp(info.logprob):.3f}".ljust(8) for info in chunk)
                    print("Probs:   ", prob_line)
                    
                    # Print tokens line
                    token_line = " ".join(str(info.token).ljust(8) for info in chunk)
                    print("Tokens:  ", token_line)
                    print()
            else:
                print(result.response_text)
            
            print("-" * 50)

def main():
    # Example usage
    playground = ChatGPTPlayground()
    
    # Test parameters
    prompt = "What is artificial intelligence?"
    parameters = {
        "temperature": 1,
        "top_p": 1,
        "seed": None,
        "n_completions": 5,
        "max_tokens": 100
    }
    
    # Generate completions
    results = playground.generate_completions(
        prompt=prompt,
        n_completions=parameters["n_completions"],
        temperature=parameters["temperature"],
        top_p=parameters["top_p"],
        seed=parameters["seed"],
        max_tokens=parameters["max_tokens"]
    )
    
    # Print and save results
    playground.print_results(results)
    # playground.save_results(results, parameters)

if __name__ == "__main__":
    main()
