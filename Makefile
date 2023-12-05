create_folders_in_cloud:
	@mkdir ~/.lewagon/saved_models

reload_api:
	uvicorn pseudoproof.fast.api:app --reload

docker_push:
	docker push europe-southwest1-docker.pkg.dev/wagon-bootcamp-barna-401510/pseudoproof/pseudoproof_image

docker_build:
	docker build -t europe-southwest1-docker.pkg.dev/wagon-bootcamp-barna-401510/pseudoproof/pseudoproof_image:latest .

gcloud_deploy:
	gcloud run deploy --image europe-southwest1-docker.pkg.dev/wagon-bootcamp-barna-401510/pseudoproof/pseudoproof_image:latest  --region europe-west1 --env-vars-file .env.yaml

streamlit:
	python -m streamlit run streamlit/home.py
