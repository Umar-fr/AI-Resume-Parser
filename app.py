from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rank import write_submission


def run_ranker(candidates_path: str | Path, out_path: str | Path, limit: int = 100):
    candidates_path = Path(candidates_path)
    out_path = Path(out_path)

    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidates_path}")

    ranked = write_submission(candidates_path, out_path, limit=limit)
    return ranked, out_path


def main() -> None:
    st.set_page_config(page_title="Candidate Ranker Sandbox", layout="centered")
    st.title("Candidate Ranker Sandbox")
    st.write("Run the ranking pipeline from a browser and inspect the generated CSV.")

    with st.form("rank_form"):
        candidates_path = st.text_input(
            "Candidates file",
            value="data/sample_candidates.json",
            help="Use data/sample_candidates.json for a quick smoke test or data/candidates.jsonl if you have the full file.",
        )
        out_path = st.text_input(
            "Output CSV",
            value="outputs/sample_out.csv",
            help="Where the submission CSV should be written.",
        )
        limit = st.number_input(
            "Number of candidates to rank",
            min_value=1,
            max_value=1000,
            value=100,
            step=1,
        )
        submitted = st.form_submit_button("Run ranking")

    if submitted:
        try:
            ranked, output_path = run_ranker(candidates_path, out_path, limit=limit)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Ranking failed: {exc}")
            return

        st.success(f"Wrote {len(ranked)} rows to {output_path}")
        if len(ranked) < limit:
            st.info(
                "On smaller or sampled datasets, the pipeline may return fewer rows than requested. "
                "It filters out candidates that are not competitive for the role or that are flagged as honeypots, "
                "so a 50-row sample can legitimately produce only one ranked candidate."
            )

        if ranked:
            st.metric("Top candidate", ranked[0].candidate_id)
            st.metric("Top score", f"{ranked[0].rank_score:.6f}")

        if output_path.exists():
            df = pd.read_csv(output_path)
            st.subheader("Ranked results")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download CSV",
                data=output_path.read_bytes(),
                file_name=output_path.name,
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
