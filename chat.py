"""Talk to the agent by typing (text mode of the same agent).

  python chat.py            # shows reply + decision
  python chat.py --debug    # also shows tool calls and retrieved sections
Type 'reset' to start a new call, 'quit' to exit.
"""
import argparse

from src.agent import Agent
from src.llm import LLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    agent = Agent(LLM(verbose=args.debug))
    print("ZentroMart support (text). Type 'reset' for a new call, 'quit' to exit.\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() in ("quit", "exit"):
            break
        if text.lower() == "reset":
            agent.reset()
            print("-- new call --\n")
            continue
        t = agent.turn(text)
        print(f"Agent [{t['decision']}]: {t['reply']}")
        if args.debug:
            for c in t["tool_calls"]:
                print(f"   tool {c['name']}({c['args']}) -> {c['result']}")
            print(f"   retrieved: {t['retrieved']}  evidence: {t['evidence']}")
        print()


if __name__ == "__main__":
    main()
