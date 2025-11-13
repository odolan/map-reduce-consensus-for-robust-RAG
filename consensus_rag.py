# this is the map-reduce-consensus approach for robust RAG on COVID-19 questions


# sub agent - processes / summarizes a document
def doc_processor_agent(question, doc):
	
	# task: summarize output as it relates to question 
	# return response
	pass

# queries relevant documents from knowledge base
def query_relevant_docs(question):
	# task: retrieve relevant documents for question
    # return list of documents
    pass


# produces consensus response from multiple sub-agent outputs
def get_consensus_response(responses):
	# task: given list of responses, vote on consensus answer
    # return consensus answer
    pass 
	


# reducer agent - collects results from the sub agents and votes on consensus
def covid_question_agent(question):

	# relevant_docs = query_relevant_docs(question)
	# responses = [doc_processor_agent(question, doc) for doc in relevant_docs]
	pass
