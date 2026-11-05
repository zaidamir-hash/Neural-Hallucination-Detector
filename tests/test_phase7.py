"""Offline selection, alignment, evidence and export tests using synthetic examples."""
import csv
import json

import pytest

from hallucination_detector.error_analysis import (
    OUTPUT, PHASE6, PROCESSED, attach_reviews, checksum, export_phase7,
    read_json, select_errors, write_json,
)


def fixture_rows():
    examples, predictions = [], []
    for name, label, score in [('b', 0, .9), ('a', 0, .9), ('c', 0, .7),
                                ('e', 1, .1), ('d', 1, .1), ('f', 1, .2),
                                ('tn', 0, .3), ('tp', 1, .5)]:
        examples.append(dict(example_id=name, source_id='s' + name, label=label, split='test',
                             question='Where?', candidate_answer='Here',
                             reference_context='Here is the location. Extra irrelevant text.'))
        predictions.append(dict(example_id=name, source_id='s' + name, true_label=str(label),
                                continuous_score=str(score), predicted_label=str(int(score >= .5)),
                                model='mlp', seed='202'))
    return predictions, examples


def notes_for(selected):
    return [dict(example_id=r['example_id'], context_excerpts=['Here is the location.'],
                 categories=['other/unclear'], judgment='unclear judgment',
                 analyst_explanation='Synthetic analyst evidence.') for r in selected]


def test_original_model_seed_filter_and_error_definitions():
    predictions, examples = fixture_rows()
    distractors = []
    for model, seed in [('mlp', '101'), ('mlp', '303'), ('cosine', ''), ('primary_logistic', '')]:
        distractors += [dict(p, model=model, seed=seed) for p in predictions]
    rows, summary = select_errors(predictions + distractors, examples, .5)
    assert summary == dict(confusion_matrix=[[1, 3], [3, 1]], available={'FP': 3, 'FN': 3},
                           reviewed={'FP': 3, 'FN': 3})
    assert all(r['true_label'] == 0 and r['predicted_label'] == 1 for r in rows[:3])
    assert all(r['true_label'] == 1 and r['predicted_label'] == 0 for r in rows[3:])
    assert {r['seed'] for r in rows} == {202}


def test_deterministic_ranking_ties_and_distances():
    predictions, examples = fixture_rows()
    rows, _ = select_errors(predictions, examples, .5, 2)
    assert [r['example_id'] for r in rows] == ['a', 'b', 'd', 'e']
    assert select_errors(predictions[::-1], examples[::-1], .5, 2)[0] == rows
    assert [r['rank'] for r in rows] == [1, 2, 1, 2]
    assert rows[0]['score_minus_threshold'] == pytest.approx(.4)
    assert rows[2]['score_minus_threshold'] == pytest.approx(-.4)
    assert all(r['absolute_distance'] == pytest.approx(.4) for r in rows)


def test_fewer_than_ten_zero_and_exact_threshold():
    predictions, examples = fixture_rows()
    rows, summary = select_errors(predictions, examples, .5)
    assert len(rows) == 6
    assert 'tp' not in [r['example_id'] for r in rows]  # Equality is predicted 1.
    predictions = [dict(p, true_label=p['predicted_label']) for p in predictions]
    examples = [dict(e, label=int(p['true_label'])) for e, p in zip(examples, predictions)]
    assert select_errors(predictions, examples, .5)[0] == []


@pytest.mark.parametrize('fault', ['missing', 'extra', 'duplicate_prediction', 'duplicate_example',
                                  'source', 'label', 'split', 'decision', 'nan', 'score_range', 'label_range'])
def test_alignment_and_saved_decisions_fail_closed(fault):
    predictions, examples = fixture_rows()
    if fault == 'missing':
        predictions.pop()
    elif fault == 'extra':
        predictions.append(dict(predictions[0], example_id='unknown'))
    elif fault == 'duplicate_prediction':
        predictions.append(predictions[0])
    elif fault == 'duplicate_example':
        examples.append(examples[0])
    elif fault == 'source':
        predictions[0]['source_id'] = 'wrong'
    elif fault == 'label':
        examples[0]['label'] = 1
    elif fault == 'split':
        examples[0]['split'] = 'train'
    elif fault == 'decision':
        predictions[0]['predicted_label'] = '0'
    elif fault == 'nan':
        predictions[0]['continuous_score'] = 'nan'
    elif fault == 'score_range':
        predictions[0]['continuous_score'] = '1.1'
    else:
        predictions[0]['true_label'] = '-1'
    with pytest.raises(ValueError):
        select_errors(predictions, examples, .5)


@pytest.mark.parametrize('fault', ['missing_review', 'duplicate_review', 'invented_excerpt',
                                  'long_excerpt', 'category', 'judgment', 'explanation'])
def test_reviews_require_evidence_and_exact_selection(fault):
    predictions, examples = fixture_rows()
    rows, _ = select_errors(predictions, examples, .5)
    notes = notes_for(rows)
    if fault == 'missing_review':
        notes.pop()
    elif fault == 'duplicate_review':
        notes.append(notes[0])
    elif fault == 'invented_excerpt':
        notes[0]['context_excerpts'] = ['Not in the context']
    elif fault == 'long_excerpt':
        rows[0]['reference_context'] = 'a' * 651
        notes[0]['context_excerpts'] = ['a' * 651]
    elif fault == 'category':
        notes[0]['categories'] = ['mechanical invented category']
    elif fault == 'judgment':
        notes[0]['judgment'] = 'certainly false'
    else:
        notes[0]['analyst_explanation'] = ''
    with pytest.raises(ValueError):
        attach_reviews(rows, notes)


def make_export_fixture(root):
    predictions, examples = fixture_rows()
    for folder in (PHASE6, 'data/processed', 'results/metrics', OUTPUT):
        (root / folder).mkdir(parents=True, exist_ok=True)
    (root / PROCESSED).write_text(''.join(json.dumps(e) + '\n' for e in examples), encoding='utf-8')
    validation_path = root / 'results/metrics/phase5_validation.json'
    write_json(validation_path, dict(summary=dict(representative_seed=202), runs=[dict(seed=202, threshold=.5)]))
    lock_path = root / PHASE6 / 'evaluation_manifest.json'
    write_json(lock_path, dict(input_checksums={'results/metrics/phase5_validation.json': checksum(validation_path)}))
    metrics_path = root / PHASE6 / 'primary_metrics.json'
    write_json(metrics_path, dict(representative_mlp_seed=202, row_count=8,
               evaluation_manifest_checksum=checksum(lock_path), input_checksums={PROCESSED: checksum(root / PROCESSED)},
               results=[dict(model='mlp', seed=202, threshold=.5, metrics=dict(confusion_matrix=[[1, 3], [3, 1]]))]))
    prediction_path = root / PHASE6 / 'primary_predictions.csv'
    with prediction_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    write_json(root / PHASE6 / 'primary_saved.json', dict(output_checksums={
        PHASE6 + p.name: checksum(p) for p in (prediction_path, metrics_path)}))
    notes_path = root / OUTPUT / 'analyst_notes.json'
    write_json(notes_path, notes_for(select_errors(predictions, examples, .5)[0]))
    return notes_path


def test_full_export_schema_reproducibility_and_input_preservation(tmp_path):
    notes = make_export_fixture(tmp_path)
    inputs = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    manifest = export_phase7(tmp_path, notes)
    rows = read_json(tmp_path / OUTPUT / 'reviewed_cases.json')
    assert len(rows) == 6
    assert set(rows[0]) == {'example_id', 'source_id', 'error_type', 'true_label', 'predicted_label',
                            'hallucination_score', 'threshold', 'score_minus_threshold', 'absolute_distance',
                            'model', 'seed', 'question', 'candidate_answer', 'rank', 'context_excerpts',
                            'categories', 'judgment', 'analyst_explanation'}
    assert manifest['input_checksums'][PROCESSED] == checksum(tmp_path / PROCESSED)
    assert read_json(tmp_path / OUTPUT / 'category_counts.json')['categories']['other/unclear']['total'] == 6
    report = (tmp_path / OUTPUT / 'report.md').read_text(encoding='utf-8')
    assert all(r['example_id'] in report and r['analyst_explanation'] in report for r in rows)
    assert 'Extra irrelevant text.' not in report
    outputs = {p: p.read_bytes() for p in (tmp_path / OUTPUT).iterdir()}
    export_phase7(tmp_path, notes)
    assert all(p.read_bytes() == value for p, value in {**inputs, **outputs}.items())


def test_changed_original_predictions_rejected_without_outputs(tmp_path):
    notes = make_export_fixture(tmp_path)
    with (tmp_path / PHASE6 / 'primary_predictions.csv').open('a') as handle:
        handle.write('\n')
    with pytest.raises(ValueError, match='checksum mismatch'):
        export_phase7(tmp_path, notes)
    assert not (tmp_path / OUTPUT / 'reviewed_cases.json').exists()


@pytest.mark.parametrize('fault', ['processed', 'validation', 'manifest', 'matrix', 'threshold', 'representative'])
def test_provenance_and_saved_evaluation_checks(tmp_path, fault):
    notes = make_export_fixture(tmp_path)
    if fault in ('processed', 'validation', 'manifest'):
        path = tmp_path / {'processed': PROCESSED, 'validation': 'results/metrics/phase5_validation.json',
                           'manifest': PHASE6 + 'evaluation_manifest.json'}[fault]
        path.write_bytes(path.read_bytes() + b'\n')
    else:
        path = tmp_path / PHASE6 / 'primary_metrics.json'
        metrics = read_json(path)
        if fault == 'matrix':
            metrics['results'][0]['metrics']['confusion_matrix'] = [[2, 2], [3, 1]]
        elif fault == 'threshold':
            metrics['results'][0]['threshold'] = .6
        else:
            metrics['representative_mlp_seed'] = 101
        write_json(path, metrics)
        saved_path = tmp_path / PHASE6 / 'primary_saved.json'
        saved = read_json(saved_path)
        saved['output_checksums'][PHASE6 + path.name] = checksum(path)
        write_json(saved_path, saved)
    with pytest.raises(ValueError):
        export_phase7(tmp_path, notes)
    assert not (tmp_path / OUTPUT / 'reviewed_cases.json').exists()
