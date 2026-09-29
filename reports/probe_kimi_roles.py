"""Separate-pilot native-schema check; gaps never choose a provider."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import (
    ContractError,
    common_parser,
    digest,
    file_hash,
    generate_rubric,
    initialize_run,
    main_guard,
    operational_summary,
    parse_args,
    propose_prompt,
    read_json,
    role_config,
    run_lock,
    selected_examples,
    write_json,
)


def main():
    parser = common_parser(__doc__, ['rubric', 'optimizer'])
    parser.set_defaults(split='pilot')
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--rubric-example-id', help='Explicit operational test case from the pilot training paper')
    parser.add_argument('--max-meta-prompt-words', type=int, default=800)
    args = parse_args(parser)
    if args.split != 'pilot':
        raise ContractError('Capability checks use only separate pilot papers')
    source = Path(args.source_run)
    manifest = read_json(source / 'manifest.json')
    if manifest['arguments']['split'] != 'pilot' or file_hash(args.dataset) != manifest['dataset_hash']:
        raise ContractError('Probe source must be the same frozen separate pilot dataset')
    feedback = read_json(source / 'feedback/iter_01/training.json')
    if not feedback['summary']['complete']:
        raise ContractError('Full pilot training baseline required for the optimizer probe')
    examples = selected_examples(args)
    paper = min(e['paper_id'] for e in examples)
    train = [{**e, 'split': 'train'} for e in examples if e['paper_id'] == paper]
    example = min(train, key=lambda e: e['example_id'])
    if args.rubric_example_id:
        matches = [e for e in train if e['example_id'] == args.rubric_example_id]
        if len(matches) != 1:
            raise ContractError('Explicit probe example must belong to the fixed pilot training paper')
        example = matches[0]
    initial = (source / 'prompts/iter_00.md').read_text()
    roles = {role: role_config(args, role) for role in ('rubric', 'optimizer')}
    with run_lock(args.output_dir):
        api = initialize_run(args, roles, 'role_capability_probe', {
            'source_hash': manifest['substantive_hash'], 'feedback_hash': digest(feedback),
            'initial_prompt_hash': digest(initial), 'probe_script_hash': file_hash(__file__),
            'gate_uses_score_gap': False,
            'example_selection': 'explicit_operational_case' if args.rubric_example_id else 'first_sorted_pilot_training_example',
            'example_id': example['example_id']})
        rubric = generate_rubric(api, example, initial, {'probe': 'rubric'},
                                Path(args.output_dir) / 'rubrics/probe.json')
        propose_prompt(api, initial, feedback, train, initial, args, 1)
        proposal = read_json(Path(args.output_dir) / 'feedback/iter_01/proposal.json')
        format_valid = rubric['status'] == 'valid' and any(a['status'] == 'valid' for a in proposal['attempts'])
        write_json(Path(args.output_dir) / 'status.json', {
            'state': 'complete' if format_valid else 'format_incomplete', 'rubric_status': rubric['status'],
            'optimizer_json_valid': any(a['status'] == 'valid' for a in proposal['attempts']),
            'proposal_accepted': proposal['accepted'], 'gate_uses_score_gap': False, 'paid': True})
        operational_summary(args.output_dir)
        write_json(Path(args.output_dir) / 'costs.json', api.ledger.summary())
        print('Native schema probe complete; format_valid:', format_valid, flush=True)


if __name__ == '__main__':
    main_guard(main)
