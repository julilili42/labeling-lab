from labeling_lab.link_features import LinkVerdictInput, make_text


def test_link_text_uses_runtime_feature_shape():
    text = make_text(
        LinkVerdictInput(
            anchor=" City guide ",
            target_url="https://example.test/en/visit-tuebingen",
            parent_url="https://origin.test/",
            parent_depth=1,
            parent_pageverdict_score=0.84,
            parent_pageverdict_decision="index_strong",
            parent_relevance=8.2,
        )
    )

    assert "anchor: City guide" in text
    assert "target_path: en visit tuebingen" in text
    assert "parent_pageverdict_score_bucket: 0.80" in text
    assert "flags: hard_skipable_url:no" in text
