import pandas as pd
import anthropic
import json
import os

client = anthropic.Anthropic(api_key="INSERT_KEY")

SCORING_RULES = """
You are an expert data annotator for skincare reviews. 
Score the provided reviews based STRICTLY on the following detailed rules and logic gates. 

--- MACRO-ASPECTS ---
Output exactly one of the following strings for each macro-aspect: "+1", "-1", "0", or "NaN".

Summary Rule:
* "+1" -> strictly positive: Only positive sentiment about the aspect. No minor/implied negatives.
* "-1" -> strictly negative: Only negative sentiment about the aspect. No minor/implied positives.
* "0"  -> mixed or neutral sentiment: Any presence of both positive and negative sentiment, or purely neutral statements.
* "NaN" -> no mention / opinion not given.

1. "sensoriality" (Sensoriality)
* Definition: Immediate tactile and olfactory experience during/after application. Strictly how it feels (tactile) and smells (olfactory).
* Exclusions: NOT Aesthetic (shiny/matte belongs in finish). NOT Hydration Status (tight/dry is a functional failure -> performance). 
* Positive Examples: "Absorbs instantly," "lightweight", "unscented", "smooth texture".
* Negative Examples: "Feels sticky", "smells like chemicals", "heavy on the skin", "pills when rubbing".

2. "performance" (Performance)
* Definition: Explicit functional, long-term biological or cosmetic outcome (hydration, anti-aging, barrier maintenance, melanin-based tone correction like fading dark spots).
* Exclusions: NOT General Quality ("amazing", "best"). NOT Sensorial Action ("sinks in" is delivery, not result). NOT Aesthetic ("instant glow" is temporary/finish, but "restored radiance over a month" is performance). NOT Inflammatory Tone/Safety (fading hyperpigmentation is performance; soothing rosacea/allergic redness is safety). NOT Medical Healing (repairing moisture barrier is performance; healing bleeding eczema is safety).
* The Tests: 
  - Specificity Test: Does it name a skin benefit? If no -> NaN.
  - "So What?" Test: If "it penetrates deeply," ask "So what?" If no answer regarding hydration/plumpness, it remains Sensoriality.
* Positive Examples: "Keeps skin hydrated all day", "plumped fine lines", "repaired my skin barrier".
* Negative Examples: "Dried me out by noon", "did nothing for my wrinkles".

3. "finish" (Aesthetic Finish)
* Definition: Immediate visual appearance and optical effect on the skin (light reflection, color alteration, surface finish).
* Exclusions: NOT Sensoriality ("feels greasy" is tactile/sensoriality; "looks greasy" is visual/finish). NOT Performance ("faded dark spots" is biological/performance; "instant tone-up" is finish). NOT Safety (redness is biological/safety, not an aesthetic finish).
* Positive Examples: "Gives me glass skin", "perfect matte finish", "dewy glow", "leaves no white cast", "blurs my pores visually".
* Negative Examples: "Leaves a terrible white cast", "makes me look like a greaseball", "too shiny", "dull finish", "looks chalky".

4. "safety" (Safety)
* Definition: Any dermatological, immune, medical, or inflammatory reaction. Covers acne, breakouts, inflammatory redness, stinging, burning, allergic reactions, eczema, rosacea, and sensitive skin tolerance.
* Exclusions: NOT Cosmetic failures ("didn't fix wrinkles" is performance). NOT Melanin/Pigment ("faded dark spots" is performance). NOT Cosmetic Moisture ("didn't hydrate" is performance; "made my skin painfully dry, cracked, and burning" is safety).
* Positive Examples: "Cleared my breakouts", "calmed my eczema", "no stinging", "good for sensitive skin", "Redness subsided after using this cream".
* Negative Examples: "Gave me cystic acne", "burned my skin", "triggered rosacea".

5. "extrinsic" (Extrinsic)
* Definition: Factors external to the chemical formula. Strictly limited to price, value, packaging mechanics, and container hygiene.
* Exclusions: NOT Logistics (shipping/transit/courier). NOT Transit Damage (if it arrived broken due to shipping, this is Noise/NaN).
* Positive Examples: "Great value", "hygienic pump", "luxurious heavy jar".
* Negative Examples: "Way too expensive", "pump broke", "unhygienic packaging".

--- BINARY VARIABLES ---
Output strictly as integers: 1 (True) or 0 (False).

6. "mentions_ingredient" (Mentions Ingredient)
* Rule: Score 1 ONLY if explicitly naming a recognized chemical compound, plant extract, or standardized cosmetic category (e.g., Niacinamide, hyaluronic acid, ceramides, fragrance, alcohol, parabens, silicones). Includes "free from" claims.
* Excludes (Score 0): Vague descriptors ("chemicals", "natural stuff", "toxins", "water").

7. "mentions_routine" (Mentions Routine Word)
* Rule: Score 1 if explicitly describing placement/timing in a regimen OR compatibility with other products.
* Includes: "Step", "routine", "layering", "AM", "PM", "after my toner", "doesn’t clash with my sunscreen", "works with my serum".
* Excludes (Score 0): NOT Makeup (makeup compatibility goes to mentions_makeup). NOT casual time passing ("used it yesterday"). NOT generic product mentions without usage context.

8. "mentions_makeup" (Mentions Makeup)
* Rule: Score 1 ONLY if explicitly mentioning color cosmetics applied over or under the moisturizer.
* Includes: "Foundation", "concealer", "primer", "BB cream", "makeup", "makes my foundation separate".
* Excludes (Score 0): Sunscreen/SPF (unless tinted), lip balm, other skincare.

9. "mentions_korea" (Mentions Korea)
* Rule: Score 1 if mentioning Korea in any way.
* Examples: "I love kbeauty", "Korean skincare quality", "Popular in Korea".

--- UNSPECIFIED SENTIMENT ---
10. "unspecified_sentiment"
* Output string: "+1", "-1", "0", or "NaN".
* Logic Gate 1: Check Macro-Aspects. IF ANY of the 5 Macro-Aspects (sensoriality, performance, finish, safety, extrinsic) are scored "+1", "-1", or "0", THEN unspecified_sentiment MUST be "0".
* Logic Gate 2: IF ALL 5 Macro-Aspects are "NaN", evaluate the general tone:
  - "+1": General praise ("Love it", "My holy grail").
  - "-1": General dislike ("Hate it", "Worst ever").
  - "0": Neutral/Mixed ("I like it, but...", "Normal cream")
  - "NaN": Not a product review (e.g., only about shipping, random sentences).

--- OUTPUT FORMAT ---
Respond with a raw JSON ARRAY of objects. Do not omit any reviews. Do not use markdown blocks like ```json in the output, just the raw JSON text.
[
  {
    "ID": <integer from input>,
    "sensoriality": "",
    "performance": "",
    "finish": "",
    "safety": "",
    "extrinsic": "",
    "mentions_ingredient": 0,
    "mentions_routine": 0,
    "mentions_makeup": 0,
    "mentions_korea": 0,
    "unspecified_sentiment": ""
  }
]
"""

BATCH_SIZE = 5

#### Building the batch requests ####
def build_batch_requests(df, batch_size=BATCH_SIZE):
    requests = []

    for i in range(0, len(df), batch_size):
        df_chunk = df.iloc[i:i + batch_size]

        prompt_data = []
        for _, row in df_chunk.iterrows():
            prompt_data.append({
                "ID": int(row["ID"]),
                "text": str(row["review_text"])
            })

        prompt_text = f"Score this batch:\n{json.dumps(prompt_data, ensure_ascii=False)}"

        requests.append({
            "custom_id": f"chunk_{i}",          
            "params": {
                "model": "claude-sonnet-4-6",
                "max_tokens": 4096,
                "system": [
                  {
                    "type": "text",
                    "text": SCORING_RULES,
                    "cache_control": {"type": "ephemeral"}
                    }
                    ],
                "messages": [
                    {"role": "user", "content": prompt_text}
                ]
            }
        })

    return requests

def submit_batch(requests):
    print(f"Sening batch with {len(requests)} API requestes...")
    batch = client.messages.batches.create(requests=requests)
    print(f"Done!\n")
    print(f"==========================================")
    print(f"Batch ID: {batch.id}")
    print(f"==========================================\n")
    return batch

if __name__ == "__main__":
    try:
      input_file = "data/01_interim/data.xlsx"
    # if full file doesnt exist, try with sample_data.xlsx
    except Exception as e:
      print(f"Error: {e}")
      input_file = "data/01_interim/sample_data.xlsx"

    if not os.path.exists(input_file):
        print(f"Error: Couldn't find the file {input_file}")
        exit()

    df = pd.read_excel(input_file, engine="openpyxl")
    print(f"Loading {len(df)} reviews from {input_file}")

    # Constructing the batch requests
    requests = build_batch_requests(df, batch_size=BATCH_SIZE)

    # Sending the batch to the API
    batch = submit_batch(requests)

    # Saving the batch ID to a file for later retrieval
    with open("batch_id.txt", "w") as f:
        f.write(batch.id)
    
    print("Batch ID saved!")
