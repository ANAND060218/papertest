import os
import sys
from compare_apis import run_api_comparison_and_store_samples


def main():
    print("""
====================================================================
  KYC-FREE INDIAN STOCK MARKET DATA & ML SUITE
====================================================================
 Evaluating 100% free, KYC-free options for Indian Stock Market data
 (NSE/BSE) to build and train Machine Learning models without opening
 a Demat account or submitting KYC documents.
====================================================================
""")
    run_api_comparison_and_store_samples()


if __name__ == "__main__":
    main()
