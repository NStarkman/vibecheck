import numpy as np
import pandas as pd
import pytest

from vibecheck import Settings, Vibecheck
from vibecheck.analysis import AgeModel, estimate_mood, quadrant, roast
from vibecheck.analysis.wellbeing import valence_trend
from vibecheck.demo import demo_age_samples, demo_history, demo_labelled_playlists, demo_playlist
from vibecheck.ml import build_dataset, title_label, train_mood_classifier
from vibecheck.nlp.emotion import LexiconEmotion, emotions_to_va, looks_english, text_stats


@pytest.fixture(scope="module")
def vc():
    return Vibecheck(Settings(), offline=True, verbose=False)


@pytest.fixture(scope="module")
def profile(vc):
    df = demo_playlist()
    return vc.analyze(df, meta={"name": "t"}, history=demo_history(df), tz="America/New_York")


def test_quadrants():
    assert quadrant(0.8, 0.8) == "Euphoric" and quadrant(0.2, 0.8) == "Angsty"
    assert quadrant(0.2, 0.2) == "Melancholy" and quadrant(0.8, 0.2) == "Mellow" and quadrant(0.51, 0.49) == "Balanced"


def test_mood_ci_contains_mean_and_weights_matter():
    df = pd.DataFrame({"mood_valence": [0.1, 0.9], "mood_arousal": [0.5, 0.5]})
    even = estimate_mood(df)
    assert even.valence_ci[0] <= even.valence <= even.valence_ci[1]
    skew = estimate_mood(df, weights=pd.Series([9.0, 1.0]))
    assert skew.valence < even.valence


def test_emotion_va_mapping_direction():
    sad = dict(anger=0, disgust=0, fear=0, joy=0, neutral=0.2, sadness=0.8, surprise=0)
    joy = dict(anger=0, disgust=0, fear=0, joy=0.8, neutral=0.2, sadness=0, surprise=0)
    assert emotions_to_va(sad)[0] < 0.5 < emotions_to_va(joy)[0]


def test_lexicon_model_and_text_stats():
    text = "I cry alone and the tears won't stop\nI miss you\nI miss you\n" * 3
    p = LexiconEmotion().score([text])[0]
    assert p["sadness"] > p["joy"]
    stats = text_stats(text)
    assert stats["repetition"] > 0.5 and stats["first_person"] > 0.1
    assert looks_english(text) and not looks_english("dans la nuit je pense à toi mon amour " * 5)


def test_age_model_recovers_planted_age():
    m = AgeModel(now_year=2026)
    rng = np.random.default_rng(0)
    years = np.round(rng.normal(2026 - 30 + 17, 3, 40))  # a 30-year-old's formative-years music
    est = m.estimate(years)
    assert est.interval80[0] <= 30 <= est.interval80[1]
    assert abs(est.age - 30) <= 3


def test_age_fit_improves_log_loss():
    train, test = demo_age_samples(40, seed=1), demo_age_samples(30, seed=2)
    m = AgeModel()
    before = m.score(test)["log_loss"]
    m.fit(train, bump_ages=(15, 17), bump_sds=(4, 5), tempering_ns=(12, 25))
    assert m.score(test)["log_loss"] <= before


def test_profile_end_to_end(profile):
    assert profile.mood.label in {"Euphoric", "Angsty", "Melancholy", "Mellow", "Balanced"}
    assert profile.age is not None and 13 <= profile.age.age <= 75
    assert 0 <= profile.personality.craft_score <= 5
    assert profile.roast and all(isinstance(r, str) for r in profile.roast)
    assert "not a read on your mental health" in profile.wellbeing.disclaimer
    md = profile.to_markdown()
    assert "## t:" in md and "### The roast" in md


def test_history_behavior_and_trend(profile):
    b = profile.behavior
    assert 0 <= b.night_share <= 1 and 0 <= b.skip_rate <= 1 and b.n_plays > 1000
    assert abs(b.hour_share.sum() - 1) < 1e-9
    assert profile.wellbeing.trend is not None and profile.wellbeing.trend["z"] < 0  # demo plants a recent dip


def test_valence_trend_needs_enough_data():
    t = pd.DataFrame({"valence": [0.5] * 5, "coverage": [1.0] * 5})
    assert valence_trend(t) is None


def test_roast_never_empty_and_formats():
    assert roast({})  # falls back to the "nothing to roast" line
    lines = roast({"top_artist_share": 0.4, "top_artist": "Someone", "n_tracks": 30})
    assert any("Someone" in line for line in lines)


@pytest.mark.parametrize("title,label", [
    ("sad girl autumn", "sad"), ("GYM 🔥", "hype"), ("lofi study", None), ("love songs", "romantic"),
    ("songs i love", None), ("random", None),
])
def test_title_labels(title, label):
    assert title_label(title) == label


def test_weak_label_classifier_beats_baseline(vc):
    pls = demo_labelled_playlists(per_mood=3, tracks_per=15)
    X, y, g, used = build_dataset([(t, vc.enrich(df)) for t, df in pls])
    clf = train_mood_classifier(X, y, g, n_splits=3)
    assert clf.cv["macro_f1"] > clf.cv["baseline_macro_f1"] + 0.3
    probs = clf.predict_playlist(pls[0][1].pipe(vc.enrich))
    assert abs(probs.sum() - 1) < 1e-6
