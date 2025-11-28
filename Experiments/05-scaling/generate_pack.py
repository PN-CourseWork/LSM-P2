#!/usr/bin/env python3
"""
Generate LSF job pack for scaling experiments.

Usage:
    python generate_pack.py --type strong --output jobs.pack
    python generate_pack.py --type weak
"""

import argparse
from pathlib import Path
import sys
from utils.hpc import load_config, generate_pack_lines, write_pack_file

def main():
    parser = argparse.ArgumentParser(description="Generate LSF job pack")
    parser.add_argument("--type", choices=["strong", "weak"], required=True, help="Scaling type")
    parser.add_argument("--output", type=str, default="jobs.pack", help="Output pack file")
    # Optional override for strong scaling grid sizes
    parser.add_argument("--N", type=int, nargs="+", help="Grid sizes for strong scaling")
    
    args = parser.parse_args()
    
    # Load config based on type
    config_path = Path(__file__).parent / f"{args.type}_scaling.yaml"
    if not config_path.exists():
        print(f"Error: Config file {config_path} not found")
        sys.exit(1)
        
    config = load_config(config_path)
    
    # Override N if provided
    if args.N and args.type == "strong":
        if "groups" in config:
            for group in config["groups"]:
                if "sweep" in group:
                    group["sweep"]["N"] = args.N
        elif "sweep" in config:
            config["sweep"]["N"] = args.N
        
    # Generate content
    job_name_base = f"{args.type}_scaling"
    lines = generate_pack_lines(config, job_name_base)
    
    # Write file
    write_pack_file(args.output, lines)
            
    print(f"Generated {len(lines)} jobs in {args.output}")

if __name__ == "__main__":
    main()