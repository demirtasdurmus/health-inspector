from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from health_inspector.inspection import load_database, get_vision_observation, search_laws, run_judge

def main():
    db=load_database()

    print("-"*50)

    vision_observation=get_vision_observation()

    print("-"*50)

    matched_laws=search_laws(db, vision_observation)

    print("-"*50)

    if not matched_laws:
        print("❌ No results found for the vision observation.")
        exit(1)
    else:
        verdict = run_judge(vision_observation, matched_laws)

        print("=" * 50)
        print("INSPECTION REPORT:")
        print(verdict)
        print("=" * 50)




if __name__ == "__main__":
    main()
