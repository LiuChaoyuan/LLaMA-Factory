# main.py
import argparse
import os
import config # To ensure OUTPUT_DIR is accessible for path creation

# These imports are needed if you call functions directly from main
# from train import train_model 
# from predict import predict_on_test_set 

def main():
    parser = argparse.ArgumentParser(description="Chinese Hate Speech Quadruplet Extraction Task")
    parser.add_argument("--do_train", action="store_true", help="Run training.")
    parser.add_argument("--do_predict", action="store_true", help="Run prediction on test set(s).")
    parser.add_argument("--test_file", type=str, default="test1", choices=["test1", "test2", "both"],
                        help="Which test file to predict on: test1, test2, or both. Default is test1.")
    
    # You can add more arguments to override config.py settings if needed
    # parser.add_argument("--model_name", type=str, help="Override model name from config.")
    # parser.add_argument("--num_epochs", type=int, help="Override number of training epochs.")

    args = parser.parse_args()

    if args.do_train:
        print("--- Starting Training Phase ---")
        # Dynamically import train_model to avoid loading heavy libraries if only predicting
        from train import train_model 
        
        # Ensure output directories exist
        if not os.path.exists(config.OUTPUT_DIR):
            os.makedirs(config.OUTPUT_DIR)
        checkpoint_dir = os.path.join(config.OUTPUT_DIR, "training_checkpoints")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        log_dir = os.path.join(config.OUTPUT_DIR, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        train_model()
        print("--- Training Finished ---")

    if args.do_predict:
        print("--- Starting Prediction Phase ---")
        # Dynamically import predict_on_test_set
        from predict import predict_on_test_set

        trained_model_path = os.path.join(config.OUTPUT_DIR, "best_model_after_eval")
        if not os.path.exists(trained_model_path):
            print(f"Error: Trained model not found at {trained_model_path}. Cannot run prediction.")
            return

        if args.test_file == "test1" or args.test_file == "both":
            print(f"Predicting on {config.TEST1_FILE}...")
            predict_on_test_set(
                model_path=trained_model_path,
                test_file_path=config.TEST1_FILE,
                output_file_path=os.path.join(config.OUTPUT_DIR, "predictions_test1.json")
            )
        
        if args.test_file == "test2" or args.test_file == "both":
            if os.path.exists(config.TEST2_FILE):
                print(f"Predicting on {config.TEST2_FILE}...")
                predict_on_test_set(
                    model_path=trained_model_path,
                    test_file_path=config.TEST2_FILE,
                    output_file_path=os.path.join(config.OUTPUT_DIR, "predictions_test2.json")
                )
            else:
                print(f"Warning: {config.TEST2_FILE} not found, skipping prediction for it.")
        print("--- Prediction Finished ---")

    if not args.do_train and not args.do_predict:
        print("No action specified. Use --do_train or --do_predict.")
        parser.print_help()

if __name__ == "__main__":
    main()