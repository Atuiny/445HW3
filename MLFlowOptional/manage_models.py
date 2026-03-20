"""
Model Registry and Lifecycle Management Script
This script demonstrates MLflow Model Registry operations for Part E

LEARNING: This script shows how to programmatically manage model versions
and their lifecycle stages (None, Staging, Production, Archived) using
the MLflow tracking client API.
"""
import argparse
from pathlib import Path
import sys

import mlflow
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_TRACKING_URI = f"sqlite:///{(REPO_ROOT / 'MLFlowOptional' / 'mlflow.db').as_posix()}"

mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)

# LEARNING: Initialize MLflow client for interacting with the tracking server
# The client allows us to query and modify model registry information
client = MlflowClient()


def list_model_versions(model_name):
    """
    List all versions of the registered model.
    
    LEARNING: This function demonstrates how to:
    1. Search for all versions of a registered model
    2. Display version metadata (stage, run_id, description)
    
    Args:
        model_name: Name of the registered model to query
    """
    print(f"\n{'='*60}")
    print(f"All versions of '{model_name}':")
    print(f"{'='*60}")
    
    try:
        # LEARNING: Search for all versions of our model using a filter query
        # This returns a list of ModelVersion objects
        versions = client.search_model_versions(f"name='{model_name}'")
        
        if not versions:
            print(f"\nNo versions found for model '{model_name}'")
            return
        
        # LEARNING: Iterate through each version and display its metadata
        for version in versions:
            print(f"\nVersion: {version.version}")
            print(f"  Stage: {version.current_stage}")  # Current lifecycle stage
            print(f"  Run ID: {version.run_id}")  # Original training run
            print(f"  Description: {version.description}")  # User-added notes
    except Exception as e:
        print(f"Error: {e}")
        print(f"Model '{model_name}' may not exist yet. Run train.py with --register_model first.")


def transition_model_stage(model_name, version, stage, description=""):
    """
    Transition a model version to a new stage.
    
    LEARNING: Model lifecycle stages help organize models:
    - None: Default state, no specific stage assigned
    - Staging: Model is being tested/validated before production
    - Production: Model is actively deployed and serving predictions
    - Archived: Deprecated model, no longer in use
    
    Args:
        model_name: Name of the registered model
        version: Model version number (1, 2, etc.)
        stage: One of 'Staging', 'Production', 'Archived', or 'None'
        description: Comment explaining the transition
    """
    print(f"\n{'='*60}")
    print(f"Transitioning model '{model_name}' version {version} to {stage}")
    print(f"{'='*60}")
    
    try:
        # LEARNING: Transition the model version to a new lifecycle stage
        # archive_existing_versions=False means other versions keep their current stage
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=False  # Keep other versions in their current stage
        )
        
        # LEARNING: Add or update the description for this model version
        # Descriptions help document why a model was transitioned
        if description:
            client.update_model_version(
                name=model_name,
                version=version,
                description=description
            )
        
        print(f"✓ Successfully transitioned version {version} to {stage}")
        if description:
            print(f"✓ Added description: {description}")
            
    except Exception as e:
        print(f"✗ Error: {e}")


def main():
    """
    Main function to demonstrate model registry lifecycle management.
    
    LEARNING: This shows a typical workflow:
    1. Parse command-line arguments for flexibility
    2. List current model versions
    3. Optionally transition models to different stages
    4. List versions again to verify changes
    """
    # LEARNING: Parse command-line arguments to make the script dynamic
    parser = argparse.ArgumentParser(
        description='Manage MLflow Model Registry lifecycle stages',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all versions of a model
        python MLFlowOptional/manage_models.py --model adult-logistic-regression-model --list
  
  # Transition version 1 to Staging
        python MLFlowOptional/manage_models.py --model adult-logistic-regression-model --version 1 --stage Staging --description "Testing phase"
  
  # Transition version 2 to Production
        python MLFlowOptional/manage_models.py --model adult-random-forest-model --version 1 --stage Production --description "Deployed model"
        """
    )
    
    parser.add_argument('--model', type=str, default='adult-logistic-regression-model',
                        help='Name of the registered model (default: adult-logistic-regression-model)')
    parser.add_argument('--list', action='store_true',
                        help='List all versions of the model')
    parser.add_argument('--version', type=int,
                        help='Model version number to transition')
    parser.add_argument('--stage', type=str, choices=['Staging', 'Production', 'Archived', 'None'],
                        help='Stage to transition to: Staging, Production, Archived, or None')
    parser.add_argument('--description', type=str, default='',
                        help='Description/comment for the transition')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("MLflow Model Registry Management")
    print("="*60)
    print(f"Tracking URI: {DEFAULT_TRACKING_URI}")
    
    # LEARNING SECTION 1: View current state of all model versions
    # This shows us what versions exist and their current stages
    list_model_versions(args.model)
    
    # LEARNING SECTION 2: Perform transition if specified
    if args.version is not None and args.stage is not None:
        transition_model_stage(
            model_name=args.model,
            version=args.version,
            stage=args.stage,
            description=args.description
        )
        
        # LEARNING SECTION 3: Verify the transition was successful
        # List versions again to see the updated stages and descriptions
        list_model_versions(args.model)
    elif args.version is not None or args.stage is not None:
        print("\n⚠ Error: Both --version and --stage must be specified together")
    
    # LEARNING: Show helpful instructions if just listing
    if args.list or (args.version is None and args.stage is None):
        print("\n" + "="*60)
        print("Usage Examples:")
        print("="*60)
        print("1. Transition a model version to Staging:")
        print(f"   python MLFlowOptional/manage_models.py --model {args.model} --version 1 --stage Staging --description 'Testing'")
        print("")
        print("2. Transition a model version to Production:")
        print(f"   python MLFlowOptional/manage_models.py --model {args.model} --version 2 --stage Production --description 'Deployed'")
        print("")
        print("3. View MLflow UI:")
        print("   mlflow ui --backend-store-uri sqlite:///MLFlowOptional/mlflow.db --port 5000")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
