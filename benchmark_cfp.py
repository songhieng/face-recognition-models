import os
import argparse
import pandas as pd
from models.elasticface import load_model
from models.elasticface.benchmark import benchmark, evaluate_protocol_splits

def main():
    parser = argparse.ArgumentParser(description='Benchmark ElasticFace on CFP dataset')
    parser.add_argument('--backbone', type=str, default='pretrain-model/295672backbone.pth', 
                        help='Path to backbone checkpoint')
    parser.add_argument('--data_dir', type=str, default='cfp-dataset/Data/Images', 
                        help='Path to CFP dataset images')
    parser.add_argument('--protocol_dir', type=str, default='cfp-dataset/Protocol', 
                        help='Path to CFP protocol directory')
    parser.add_argument('--threshold', type=float, default=0.3, 
                        help='Similarity threshold for verification')
    parser.add_argument('--mode', type=str, choices=['benchmark', 'protocol'], default='protocol', 
                        help='Evaluation mode: benchmark (all individuals) or protocol (standard CFP protocol)')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Check if files and directories exist
    if not os.path.exists(args.backbone):
        print(f"Error: Backbone file not found at {args.backbone}")
        return
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found at {args.data_dir}")
        return
    
    if args.mode == 'protocol' and not os.path.exists(args.protocol_dir):
        print(f"Error: Protocol directory not found at {args.protocol_dir}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading ElasticFace model...")
    try:
        model = load_model(args.backbone)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run benchmark
    try:
        if args.mode == 'benchmark':
            # Run the benchmark on all individuals
            print(f"Running benchmark on {args.data_dir}")
            all_results = benchmark(args.data_dir, model, args.threshold)
            
            # Aggregate overall averages
            df = pd.DataFrame(all_results)
            print("\n=== Dataset-wide averages ===")
            print(f"FF accuracy: {df['ff_acc'].mean()*100:.2f}%")
            print(f"PP accuracy: {df['pp_acc'].mean()*100:.2f}%")
            print(f"FP accuracy: {df['fp_acc'].mean()*100:.2f}%")
            
            # Save results
            output_file = os.path.join(args.output_dir, "elasticface_cfp_benchmark_results.csv")
            df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        
        elif args.mode == 'protocol':
            # Evaluate using the standard CFP protocol
            print(f"Evaluating using CFP protocol from {args.protocol_dir}")
            protocol_results = evaluate_protocol_splits(args.data_dir, args.protocol_dir, model, args.threshold)
            
            # Calculate and print average results for each protocol
            for protocol_type in ['FF', 'FP']:
                protocol_df = pd.DataFrame(protocol_results[protocol_type])
                avg_accuracy = protocol_df['accuracy'].mean() * 100
                avg_auc = protocol_df['auc'].mean()
                avg_eer = protocol_df['eer'].mean()
                
                print(f"\n=== {protocol_type} Protocol Average Results ===")
                print(f"Accuracy: {avg_accuracy:.2f}%")
                print(f"AUC: {avg_auc:.4f}")
                print(f"EER: {avg_eer:.4f}")
            
            # Save results
            for protocol_type in ['FF', 'FP']:
                output_file = os.path.join(args.output_dir, f"elasticface_cfp_{protocol_type}_results.csv")
                pd.DataFrame(protocol_results[protocol_type]).to_csv(output_file, index=False)
            
            print(f"Protocol results saved to {args.output_dir}")
    except Exception as e:
        print(f"Error during benchmark: {e}")

if __name__ == '__main__':
    main() 