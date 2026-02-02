#!/usr/bin/env python
"""
GPT-based evaluation for MMHal-Bench responses.

Evaluates model responses using GPT-4 as a judge.

Usage:
    export OPENAI_API_KEY=your_key_here
    python scripts/evaluate_mmhal_gpt.py --response outputs/mmhal_baseline/mmhal_responses.json
"""

import argparse
import json
import os
import time
from openai import OpenAI
from tqdm import tqdm


EVALUATION_TEMPLATE = '''Please act as an impartial and objective judge and evaluate the quality of the response provided by a Large Multimodal Model (LMM) to the user question. Your evaluation should be mainly based on whether the response is informative, and whether the response contains any hallucination. Hallucination, in this context, refers to a situation where the LMM generates a response that includes information not present or implied in the image.

Please note that the standard human-generated answer may not be completely comprehensive. LMM's detailed analysis or reasoning should be encouraged.

To evaluate the LMM responses, first, begin your evaluation by providing a short explanation. Second, after providing your explanation, you must rate the response by choosing from the following options:
- Rating: 6, very informative with good analysis or reasoning, no hallucination
- Rating: 5, very informative, no hallucination
- Rating: 4, somewhat informative, no hallucination
- Rating: 3, not informative, no hallucination
- Rating: 2, very informative, with hallucination
- Rating: 1, somewhat informative, with hallucination
- Rating: 0, not informative, with hallucination

### Image Contents
{}

### Question
{}

### Standard Human-Generated Answer
{}

### LMM Response to Evaluate
{}
'''


def parse_args():
    parser = argparse.ArgumentParser(description="GPT-based MMHal evaluation")

    parser.add_argument("--response", type=str, required=True,
                       help="Path to model responses JSON file")
    parser.add_argument("--gpt-model", type=str, default="gpt-4",
                       help="GPT model to use for evaluation")
    parser.add_argument("--output", type=str, default=None,
                       help="Output evaluation file (default: auto-generated)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="OpenAI API key (or set OPENAI_API_KEY env var)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Get API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError(
            "OpenAI API key required. Set via --api-key or OPENAI_API_KEY environment variable"
        )

    client = OpenAI(api_key=api_key)

    # Load responses
    print(f"Loading responses from: {args.response}")
    with open(args.response, 'r') as f:
        records = json.load(f)

    print(f"Loaded {len(records)} responses")

    # Prepare output file
    if args.output:
        output_file = args.output
    else:
        output_file = args.response.replace('.json', '_gpt_eval.json')

    print(f"Output will be saved to: {output_file}")
    print(f"Using GPT model: {args.gpt_model}")
    print("\nStarting evaluation...")

    # Evaluate each response
    evaluations = []
    scores = []

    for i, record in enumerate(tqdm(records, desc="Evaluating")):
        # Prepare prompt
        image_content = ', '.join(record.get('image_content', []))
        question = record['question']
        gt_answer = record.get('gt_answer', '')
        model_answer = record['model_answer']

        input_text = EVALUATION_TEMPLATE.format(
            image_content, question, gt_answer, model_answer
        )

        # Query GPT
        response = None
        retry_count = 0
        max_retries = 3

        while response is None and retry_count < max_retries:
            try:
                response = client.chat.completions.create(
                    model=args.gpt_model,
                    messages=[
                        {"role": "user", "content": input_text}
                    ],
                    temperature=0.0,
                )
            except Exception as e:
                print(f"\nError on sample {i}: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Retrying... ({retry_count}/{max_retries})")
                    time.sleep(10)
                else:
                    print("Max retries reached, skipping sample")
                    response = None
                    break

        if response:
            eval_text = response.choices[0].message.content
            evaluations.append({
                'sample_id': i,
                'evaluation': eval_text,
                'question': question,
                'model_answer': model_answer
            })

            # Extract score
            score = None
            for s in range(7):
                if f'rating: {s}' in eval_text.lower():
                    score = s
                    break

            if score is not None:
                scores.append(score)
            else:
                print(f"\nWarning: Could not extract score from evaluation {i}")
                scores.append(0)

            time.sleep(1)  # Rate limiting

    # Compute metrics
    avg_score = sum(scores) / len(scores) if scores else 0
    hallucination_count = sum(1 for s in scores if s < 3)
    hallucination_rate = hallucination_count / len(scores) if scores else 0

    # Compute per-question-type scores (if available)
    scores_by_type = {}
    for i, record in enumerate(records[:len(scores)]):
        q_type = record.get('question_type', 'unknown')
        if q_type not in scores_by_type:
            scores_by_type[q_type] = []
        scores_by_type[q_type].append(scores[i])

    avg_by_type = {
        q_type: sum(s_list) / len(s_list)
        for q_type, s_list in scores_by_type.items()
    }

    # Print results
    print("\n" + "="*80)
    print("MMHal-Bench GPT Evaluation Results")
    print("="*80)
    print(f"Average Score: {avg_score:.2f} / 6.0")
    print(f"Hallucination Rate: {hallucination_rate:.2%}")
    print(f"Total Samples: {len(scores)}")
    print("")

    if avg_by_type:
        print("Average Score by Question Type:")
        for q_type, avg in sorted(avg_by_type.items()):
            print(f"  {q_type}: {avg:.2f}")

    print("="*80)

    # Save results
    results = {
        'evaluations': evaluations,
        'scores': scores,
        'metrics': {
            'average_score': avg_score,
            'hallucination_rate': hallucination_rate,
            'total_samples': len(scores),
            'scores_by_type': avg_by_type
        }
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Evaluation results saved to: {output_file}")


if __name__ == "__main__":
    main()
