# Package index

## Authentication

Set up and verify your Hugging Face API token.

- [`hf_set_token()`](https://farach.github.io/huggingfaceR/dev/reference/hf_set_token.md)
  : Set Hugging Face API Token
- [`hf_whoami()`](https://farach.github.io/huggingfaceR/dev/reference/hf_whoami.md)
  : Get Current Hugging Face User Information

## Text Classification

Classify text using pre-trained models or custom label sets.

- [`hf_classify()`](https://farach.github.io/huggingfaceR/dev/reference/hf_classify.md)
  : Text Classification
- [`hf_classify_zero_shot()`](https://farach.github.io/huggingfaceR/dev/reference/hf_classify_zero_shot.md)
  : Zero-Shot Classification

## Text Tasks

Summarize, translate, extract entities, and answer questions over text
and tables.

- [`hf_summarize()`](https://farach.github.io/huggingfaceR/dev/reference/hf_summarize.md)
  : Summarize Text
- [`hf_translate()`](https://farach.github.io/huggingfaceR/dev/reference/hf_translate.md)
  : Translate Text
- [`hf_ner()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ner.md)
  : Named Entity Recognition (Token Classification)
- [`hf_question_answer()`](https://farach.github.io/huggingfaceR/dev/reference/hf_question_answer.md)
  : Extractive Question Answering
- [`hf_table_question_answer()`](https://farach.github.io/huggingfaceR/dev/reference/hf_table_question_answer.md)
  : Table Question Answering

## Embeddings and Similarity

Generate vector representations and measure semantic relatedness.

- [`hf_embed()`](https://farach.github.io/huggingfaceR/dev/reference/hf_embed.md)
  : Generate Text Embeddings
- [`hf_similarity()`](https://farach.github.io/huggingfaceR/dev/reference/hf_similarity.md)
  : Compute Pairwise Similarity
- [`hf_embed_umap()`](https://farach.github.io/huggingfaceR/dev/reference/hf_embed_umap.md)
  : Dimensionality Reduction with UMAP

## Tidytext Integration

Embed, search, cluster, and extract topics from data frames.

- [`hf_embed_text()`](https://farach.github.io/huggingfaceR/dev/reference/hf_embed_text.md)
  : Embed Text with Tidytext Compatibility
- [`hf_nearest_neighbors()`](https://farach.github.io/huggingfaceR/dev/reference/hf_nearest_neighbors.md)
  : Find Nearest Neighbors by Semantic Similarity
- [`hf_cluster_texts()`](https://farach.github.io/huggingfaceR/dev/reference/hf_cluster_texts.md)
  : Cluster Texts by Semantic Similarity
- [`hf_extract_topics()`](https://farach.github.io/huggingfaceR/dev/reference/hf_extract_topics.md)
  : Extract Semantic Topics from Text

## Local Models

Optional local embeddings and classification, without hosted inference.

- [`hf_local_setup()`](https://farach.github.io/huggingfaceR/dev/reference/hf_local_setup.md)
  : Declare Optional Local Python Requirements
- [`hf_download_model()`](https://farach.github.io/huggingfaceR/dev/reference/hf_download_model.md)
  : Download a Model Snapshot for Local Inference
- [`hf_load_local_model()`](https://farach.github.io/huggingfaceR/dev/reference/hf_load_local_model.md)
  [`print(`*`<hf_local_model>`*`)`](https://farach.github.io/huggingfaceR/dev/reference/hf_load_local_model.md)
  : Load a Reusable Local Model
- [`hf_embed_local()`](https://farach.github.io/huggingfaceR/dev/reference/hf_embed_local.md)
  : Generate Embeddings with a Loaded Local Model
- [`hf_classify_local()`](https://farach.github.io/huggingfaceR/dev/reference/hf_classify_local.md)
  : Classify Text with a Loaded Local Model

## Batch Processing

Process large datasets with parallel requests and disk checkpointing.

- [`hf_embed_batch()`](https://farach.github.io/huggingfaceR/dev/reference/hf_embed_batch.md)
  : Batch Embedding Generation (In-Memory)
- [`hf_embed_chunks()`](https://farach.github.io/huggingfaceR/dev/reference/hf_embed_chunks.md)
  : Chunked Embedding Generation (Disk Checkpoints)
- [`hf_classify_batch()`](https://farach.github.io/huggingfaceR/dev/reference/hf_classify_batch.md)
  : Batch Text Classification (In-Memory)
- [`hf_classify_chunks()`](https://farach.github.io/huggingfaceR/dev/reference/hf_classify_chunks.md)
  : Chunked Text Classification (Disk Checkpoints)
- [`hf_classify_zero_shot_batch()`](https://farach.github.io/huggingfaceR/dev/reference/hf_classify_zero_shot_batch.md)
  : Batch Zero-Shot Classification (In-Memory)
- [`hf_read_chunks()`](https://farach.github.io/huggingfaceR/dev/reference/hf_read_chunks.md)
  : Read All Chunks from Directory

## Chat and Conversations

Interact with open-source language models, tools, streaming, and vision.

- [`hf_chat()`](https://farach.github.io/huggingfaceR/dev/reference/hf_chat.md)
  : LLM Chat Interface
- [`hf_tool()`](https://farach.github.io/huggingfaceR/dev/reference/hf_tool.md)
  : Define a Chat Tool
- [`hf_run_tools()`](https://farach.github.io/huggingfaceR/dev/reference/hf_run_tools.md)
  : Run Tool Calls in a Conversation
- [`hf_describe_image()`](https://farach.github.io/huggingfaceR/dev/reference/hf_describe_image.md)
  : Describe an Image
- [`hf_conversation()`](https://farach.github.io/huggingfaceR/dev/reference/hf_conversation.md)
  : Multi-turn Conversation
- [`chat()`](https://farach.github.io/huggingfaceR/dev/reference/chat.md)
  : Continue a Conversation

## Structured Output & Tools

Extract typed fields and connect model tool calls to R functions.

- [`hf_extract()`](https://farach.github.io/huggingfaceR/dev/reference/hf_extract.md)
  : Extract Structured Data from Text
- [`hf_tool()`](https://farach.github.io/huggingfaceR/dev/reference/hf_tool.md)
  : Define a Chat Tool
- [`hf_run_tools()`](https://farach.github.io/huggingfaceR/dev/reference/hf_run_tools.md)
  : Run Tool Calls in a Conversation

## Text Generation

Generate and complete text with language models.

- [`hf_generate()`](https://farach.github.io/huggingfaceR/dev/reference/hf_generate.md)
  : Text Generation
- [`hf_fill_mask()`](https://farach.github.io/huggingfaceR/dev/reference/hf_fill_mask.md)
  : Fill Mask

## Multimodal Inference

Transcribe audio, generate images, and analyze image collections.

- [`hf_transcribe()`](https://farach.github.io/huggingfaceR/dev/reference/hf_transcribe.md)
  : Transcribe Audio
- [`hf_text_to_image()`](https://farach.github.io/huggingfaceR/dev/reference/hf_text_to_image.md)
  : Generate an Image from Text
- [`hf_text_to_speech()`](https://farach.github.io/huggingfaceR/dev/reference/hf_text_to_speech.md)
  : Convert Text to Speech
- [`hf_classify_image()`](https://farach.github.io/huggingfaceR/dev/reference/hf_classify_image.md)
  : Classify Images
- [`hf_caption_image()`](https://farach.github.io/huggingfaceR/dev/reference/hf_caption_image.md)
  : Caption Images
- [`hf_detect_objects()`](https://farach.github.io/huggingfaceR/dev/reference/hf_detect_objects.md)
  : Detect Objects in Images

## Hub Discovery

Search Hub assets, inspect providers, download files, and manage
repositories.

- [`hf_default_model()`](https://farach.github.io/huggingfaceR/dev/reference/hf_default_model.md)
  : Default Models for Hugging Face Tasks
- [`hf_search_models()`](https://farach.github.io/huggingfaceR/dev/reference/hf_search_models.md)
  : Search Models on Hugging Face Hub
- [`hf_search_spaces()`](https://farach.github.io/huggingfaceR/dev/reference/hf_search_spaces.md)
  : Search Spaces on Hugging Face Hub
- [`hf_search_papers()`](https://farach.github.io/huggingfaceR/dev/reference/hf_search_papers.md)
  : Search Papers on Hugging Face
- [`hf_model_info()`](https://farach.github.io/huggingfaceR/dev/reference/hf_model_info.md)
  : Get Model Information
- [`hf_check_inference()`](https://farach.github.io/huggingfaceR/dev/reference/hf_check_inference.md)
  : Check Inference API Availability
- [`hf_list_providers()`](https://farach.github.io/huggingfaceR/dev/reference/hf_list_providers.md)
  : List Inference Providers for a Model
- [`hf_clear_provider_cache()`](https://farach.github.io/huggingfaceR/dev/reference/hf_clear_provider_cache.md)
  : Clear the Cached Inference Provider Mappings
- [`hf_list_repo_files()`](https://farach.github.io/huggingfaceR/dev/reference/hf_list_repo_files.md)
  : List Files in a Hub Repository
- [`hf_hub_download()`](https://farach.github.io/huggingfaceR/dev/reference/hf_hub_download.md)
  : Download a File from the Hub
- [`hf_create_repo()`](https://farach.github.io/huggingfaceR/dev/reference/hf_create_repo.md)
  : Create a Hub Repository
- [`hf_upload_file()`](https://farach.github.io/huggingfaceR/dev/reference/hf_upload_file.md)
  : Upload a File to a Hub Repository
- [`hf_push_dataset()`](https://farach.github.io/huggingfaceR/dev/reference/hf_push_dataset.md)
  : Push a Data Frame as a Dataset File
- [`hf_delete_repo()`](https://farach.github.io/huggingfaceR/dev/reference/hf_delete_repo.md)
  : Delete a Hub Repository
- [`hf_list_tasks()`](https://farach.github.io/huggingfaceR/dev/reference/hf_list_tasks.md)
  : List Available Tasks
- [`hf_search_datasets()`](https://farach.github.io/huggingfaceR/dev/reference/hf_search_datasets.md)
  : Search Datasets on Hugging Face Hub
- [`hf_dataset_info()`](https://farach.github.io/huggingfaceR/dev/reference/hf_dataset_info.md)
  : Get Dataset Information

## Datasets

Load, search, filter, and summarize dataset splits from the Hub.

- [`hf_load_dataset()`](https://farach.github.io/huggingfaceR/dev/reference/hf_load_dataset.md)
  : Load Dataset via Hugging Face Datasets Server API
- [`hf_dataset_search()`](https://farach.github.io/huggingfaceR/dev/reference/hf_dataset_search.md)
  : Search Dataset Rows by Text
- [`hf_dataset_filter()`](https://farach.github.io/huggingfaceR/dev/reference/hf_dataset_filter.md)
  : Filter Dataset Rows with a Condition
- [`hf_dataset_statistics()`](https://farach.github.io/huggingfaceR/dev/reference/hf_dataset_statistics.md)
  : Get Column Statistics for a Dataset Split
- [`hf_dataset_validity()`](https://farach.github.io/huggingfaceR/dev/reference/hf_dataset_validity.md)
  : Check Which Dataset Viewer Features Work for a Dataset

## Tidymodels

Use embeddings as features in tidymodels recipes.

- [`step_hf_embed()`](https://farach.github.io/huggingfaceR/dev/reference/step_hf_embed.md)
  [`tidy(`*`<step_hf_embed>`*`)`](https://farach.github.io/huggingfaceR/dev/reference/step_hf_embed.md)
  [`tunable(`*`<step_hf_embed>`*`)`](https://farach.github.io/huggingfaceR/dev/reference/step_hf_embed.md)
  : Embedding Recipe Step

## Legacy Functions

Functions from earlier versions. These may require Python/reticulate.

- [`hf_ez_conversational()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_conversational.md)
  : Create a Conversational Agent
- [`hf_ez_conversational_api_inference()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_conversational_api_inference.md)
  : Conversational API Inference
- [`hf_ez_conversational_local_inference()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_conversational_local_inference.md)
  : Conversational Local Inference
- [`hf_ez_fill_mask()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_fill_mask.md)
  : Perform Fill-in-the-Blank Tasks
- [`hf_ez_fill_mask_api_inference()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_fill_mask_api_inference.md)
  : Fill Mask API Inference
- [`hf_ez_fill_mask_local_inference()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_fill_mask_local_inference.md)
  : Fill Mask Local Inference
- [`hf_ez_question_answering()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_question_answering.md)
  : Answer Questions about a Text based on Context
- [`hf_ez_question_answering_api_inference()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_question_answering_api_inference.md)
  : Question Answering API Inference
- [`hf_ez_question_answering_local_inference()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_question_answering_local_inference.md)
  : Question Answering Local Inference
- [`hf_ez_sentence_similarity()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_sentence_similarity.md)
  : Compare Sentence Similarity Semantically
- [`hf_ez_sentence_similarity_api_inference()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_sentence_similarity_api_inference.md)
  : Sentence Similarity API Inference
- [`hf_ez_sentence_similarity_local_inference()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_sentence_similarity_local_inference.md)
  : Sentence Similarity Local Inference
- [`hf_ez_summarization()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_summarization.md)
  : Summarize a Text
- [`hf_ez_summarization_api_inference()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_summarization_api_inference.md)
  : Summarization API Inference
- [`hf_ez_summarization_local_inference()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_summarization_local_inference.md)
  : Summarization Local Inference
- [`hf_ez_table_question_answering()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_table_question_answering.md)
  : Answer Questions about a Data Table
- [`hf_ez_table_question_answering_api_inference()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_table_question_answering_api_inference.md)
  : Table Question Answering API Inference
- [`hf_ez_table_question_answering_local_inference()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_table_question_answering_local_inference.md)
  : Table Question Answering Local Inference
- [`hf_ez_text2text_generation()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_text2text_generation.md)
  : Answer General Questions
- [`hf_ez_text2text_generation_api_inference()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_text2text_generation_api_inference.md)
  : Text2Text Generation API Inference
- [`hf_ez_text2text_generation_local_inference()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_text2text_generation_local_inference.md)
  : Text2Text Generation Local Inference
- [`hf_ez_text_classification()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_text_classification.md)
  : Classify Texts into Pre-trained Categories
- [`hf_ez_text_classification_api_inference()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_text_classification_api_inference.md)
  : Text Classification API Inference
- [`hf_ez_text_classification_local_inference()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_text_classification_local_inference.md)
  : Text Classification Local Inference
- [`hf_ez_text_generation()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_text_generation.md)
  : Generate Text from a Prompt
- [`hf_ez_text_generation_api_inference()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_text_generation_api_inference.md)
  : Text Generation API Inference
- [`hf_ez_text_generation_local_inference()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_text_generation_local_inference.md)
  : Text Generation Local Inference
- [`hf_ez_token_classification()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_token_classification.md)
  : Classify parts of a Text
- [`hf_ez_token_classification_api_inference()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_token_classification_api_inference.md)
  : Token Classification API Inference
- [`hf_ez_token_classification_local_inference()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_token_classification_local_inference.md)
  : Token Classification Local Inference
- [`hf_ez_translation()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_translation.md)
  : Translate between Languages
- [`hf_ez_translation_api_inference()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_translation_api_inference.md)
  : Translation API Inference
- [`hf_ez_translation_local_inference()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_translation_local_inference.md)
  : Translation Local Inference
- [`hf_ez_zero_shot_classification()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_zero_shot_classification.md)
  : Perform Text Classification with No Context Required
- [`hf_ez_zero_shot_classification_api_inference()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_zero_shot_classification_api_inference.md)
  : Zero Shot Classification API Inference
- [`hf_ez_zero_shot_classification_local_inference()`](https://farach.github.io/huggingfaceR/dev/reference/hf_ez_zero_shot_classification_local_inference.md)
  : Zero Shot Classification Local Inference
- [`hf_load_tokenizer()`](https://farach.github.io/huggingfaceR/dev/reference/hf_load_tokenizer.md)
  : Load an AutoTokenizer from a pre-tained model
- [`hf_load_model()`](https://farach.github.io/huggingfaceR/dev/reference/hf_load_model.md)
  : Load a pre-trained AutoModel object from Hugging Face
- [`hf_load_AutoModel_for_task()`](https://farach.github.io/huggingfaceR/dev/reference/hf_load_AutoModel_for_task.md)
  : Import a pre-trained AutoModel object for a specific task.
- [`hf_load_pipeline()`](https://farach.github.io/huggingfaceR/dev/reference/hf_load_pipeline.md)
  : Load a pipeline object from Hugging Face - pipelines usually include
  a model, tokenizer and task.
- [`hf_load_sentence_model()`](https://farach.github.io/huggingfaceR/dev/reference/hf_load_sentence_model.md)
  : Load a Sentence Transformers model to extract sentence/document
  embeddings
- [`hf_list_authors()`](https://farach.github.io/huggingfaceR/dev/reference/hf_list_authors.md)
  : List Authors
- [`hf_list_datasets()`](https://farach.github.io/huggingfaceR/dev/reference/hf_list_datasets.md)
  : List Datasets
- [`hf_list_languages()`](https://farach.github.io/huggingfaceR/dev/reference/hf_list_languages.md)
  : List Languages
- [`hf_list_libraries()`](https://farach.github.io/huggingfaceR/dev/reference/hf_list_libraries.md)
  : List Libraries
- [`hf_list_licenses()`](https://farach.github.io/huggingfaceR/dev/reference/hf_list_licenses.md)
  : List Licenses
- [`hf_list_models()`](https://farach.github.io/huggingfaceR/dev/reference/hf_list_models.md)
  : List Models
- [`hf_inference()`](https://farach.github.io/huggingfaceR/dev/reference/hf_inference.md)
  : Inference using a downloaded Hugging Face model or pipeline, or
  using the Inference API
- [`hf_sentence_encode()`](https://farach.github.io/huggingfaceR/dev/reference/hf_sentence_encode.md)
  : Use a Sentence Transformers pipeline to extract
  document(s)/sentence(s) embedding(s)
- [`hf_set_device()`](https://farach.github.io/huggingfaceR/dev/reference/hf_set_device.md)
  : Try to set device to GPU for accelerated computation
- [`hf_python_depends()`](https://farach.github.io/huggingfaceR/dev/reference/hf_python_depends.md)
  : Install Python Dependencies
- [`hf_fill_mask_payload()`](https://farach.github.io/huggingfaceR/dev/reference/hf_fill_mask_payload.md)
  : Fill Mask Task Payload
- [`hf_question_answering_payload()`](https://farach.github.io/huggingfaceR/dev/reference/hf_question_answering_payload.md)
  : Question Answering Payload
- [`hf_sentence_similarity_payload()`](https://farach.github.io/huggingfaceR/dev/reference/hf_sentence_similarity_payload.md)
  : Sentence Similarity Payload
- [`hf_summarization_payload()`](https://farach.github.io/huggingfaceR/dev/reference/hf_summarization_payload.md)
  : Summarization Task Payload
- [`hf_table_question_answering_payload()`](https://farach.github.io/huggingfaceR/dev/reference/hf_table_question_answering_payload.md)
  : Table Question Answering Payload
- [`hf_text2text_generation_payload()`](https://farach.github.io/huggingfaceR/dev/reference/hf_text2text_generation_payload.md)
  : Text2Text Generation Payload
- [`hf_text_classification_payload()`](https://farach.github.io/huggingfaceR/dev/reference/hf_text_classification_payload.md)
  : Text Classification Payload
- [`hf_text_generation_payload()`](https://farach.github.io/huggingfaceR/dev/reference/hf_text_generation_payload.md)
  : Text Generation Payload
- [`hf_token_classification_payload()`](https://farach.github.io/huggingfaceR/dev/reference/hf_token_classification_payload.md)
  : Token Classification Payload
- [`hf_translation_payload()`](https://farach.github.io/huggingfaceR/dev/reference/hf_translation_payload.md)
  : Translation Payload
- [`hf_zero_shot_classification_payload()`](https://farach.github.io/huggingfaceR/dev/reference/hf_zero_shot_classification_payload.md)
  : Zero Shot Classification Payload
- [`models_with_downloads`](https://farach.github.io/huggingfaceR/dev/reference/models_with_downloads.md)
  : Dataset of model names, tasks & download info
- [`` `%>%` ``](https://farach.github.io/huggingfaceR/dev/reference/pipe.md)
  : Pipe operator
