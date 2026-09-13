import argparse
from pathlib import Path

from huggingface_hub import HfApi

REPOS = {
    'models/spam_sms': 'spam-sms-tfidf-logreg',
    'models/tweet_sentiment': 'tweet-sentiment-tfidf-logreg',
}

def main():
    parser = argparse.ArgumentParser(description='Sube los dos modelos a Hugging Face.')
    parser.add_argument('--user', required=True, help='usuario u organizacion de Hugging Face')
    parser.add_argument('--private', action='store_true')
    args = parser.parse_args()
    api = HfApi()

    for folder, name in REPOS.items():
        repo_id = f'{args.user}/{name}'
        api.create_repo(repo_id, repo_type='model', private=args.private, exist_ok=True)
        readme = Path(folder) / 'README.md'
        readme.write_text(readme.read_text().replace('<usuario>', args.user))
        api.upload_folder(folder_path=folder, repo_id=repo_id, repo_type='model')
        print(f'modelo: https://huggingface.co/{repo_id}')

if __name__ == '__main__':
    main()
