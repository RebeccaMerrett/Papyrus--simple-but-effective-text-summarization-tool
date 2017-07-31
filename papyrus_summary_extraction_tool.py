# -*- coding: utf-8 -*-

#A SIMPLE (BUT EFFECTIVE) SUMMARIZATION TOOL FOR GOVERNMENT TRANSCRIPTS

#Below is a demonstration of summary extraction tool Papyrus, which is used on a large corpus taken from a government transcript.
#This was built during a 46-hour hackathon (GovHack), so it's a quick solution, but effective in what we need it to do.

#We were utilising NLTK, but needed more time to achieve good results. 
#Online tools like Resoomer, for example, seemed to miss a lot of context with Australian government transcripts. 

#This is part of the original GovHack project, which is here: https://github.com/JesseChavez/papyrus (open source MPL 2.0 license applies)
#GovHack entry is here: https://2017.hackerspace.govhack.org/project/govdocs

#Feel free to test this script on your docs :-)
#Feel free to contact me, rebecca.merrett@gmail.com, for any feedback. I'm interested to see the results or any issues to keep improving it. 

#This will be properly integrated into the front end (http://papyrus.jessechavez.info/#/documents), and my team mate (Dany Chavez) and me will continue working on the project.
#Note: No commits to the original GovHack repo will be made until judging period is over around end of August 2017. 

#The PDF of the transcript can be found here (no. 19405): http://www.aph.gov.au/Parliamentary_Business/Hansard/Estimates_Transcript_Schedule

import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import networkx as nx

#To import the above, install these:
#pip install scikit-learn
#pip install numpy
#pip install scipy
#pip install networkx

#function_1: Quite simple, but effective in extracting useful info buried in hundreds of pages.
#Much thought went into the textual features and regular patterns that indicate or lead to useful information.
#Below is not the most elegant way to script this, LOL. I'm not a good programmer. 
#Nevertheless, this has been tested with 30 different random Senate Estimates and government speeches.

#function_2: This is a quick last-min implementation, but with more time will build on this to detect debate on heavy issues/challenges, in particular,
#and place a higher score on these types of discussions. 
#tf/idf is used with TextRank to extract interesting, topical discussions rather than extracting facts and figures.  
#Adapted from https://joshbohde.com/blog/document-summarization

def function_1(text):
	paragraphs = text.split('\n\n')
	summary = []
	for i in paragraphs:
		#This finds each regex and counts the number of times it occurs in the paragraph. We'll use this to sum the weights of each regex.
		count_1 = len(re.findall('\$.*?million', i)) #Dollar figure (big figures, not small expenditure)
		count_2 = len(re.findall('\$.*?billion', i)) #Dollar figure
		count_3 = len(re.findall('\$.*?m', i)) #Dollar figure
		count_4 = len(re.findall('\$.*?bn', i)) #Dollar figure
		count_5 = len(re.findall('[2][0][1-9][0-9]', i)) #4-digits that indicate a year from 2011 to 2099
		count_6 = len(re.findall('\%', i)) #Percentage
		count_7 = len(re.findall('per cent', i)) #Percentage
		count_8 = len(re.findall('percent', i)) #Percentage
		count_9 = len(re.findall('completed', i)) #Cue word
		count_10 = len(re.findall('complete', i)) #Cue word
		count_11 = len(re.findall('completing', i)) #Cue word
		count_12 = len(re.findall('implement', i)) #Cue word
		count_13 = len(re.findall('implementing', i)) #Cue word
		count_14 = len(re.findall('due', i)) #Cue word
		count_15 = len(re.findall('commence', i)) #Cue word
		count_16 = len(re.findall('begin work', i)) #Cue word
		count_17 = len(re.findall('invested', i)) #Cue word
		count_18 = len(re.findall('investing', i)) #Cue word
		count_19 = len(re.findall('invest', i)) #Cue word
		count_20 = len(re.findall('surplus', i)) #Keyword
		count_21 = len(re.findall('budget', i)) #Keyword
		count_22 = len(re.findall('plan', i)) #Cue word
		count_23 = len(re.findall('planning', i)) #Cue word
		sum_1 = 1.2 * count_1
		sum_2 = 1.2 * count_2
		sum_3 = 1.2 * count_3
		sum_4 = 1.2 * count_4
		sum_5 = 0.8 * count_5
		sum_6 = 0.6 * count_6
		sum_7 = 0.6 * count_7
		sum_8 = 0.6 * count_8
		sum_9 = 0.5 * count_9
		sum_10 = 0.5 * count_10
		sum_11 = 0.5 * count_11
		sum_12 = 0.5 * count_12
		sum_13 = 0.5 * count_13
		sum_14 = 0.5 * count_14
		sum_15 = 0.5 * count_15
		sum_16 = 0.5 * count_16
		sum_17 = 0.5 * count_17
		sum_18 = 0.5 * count_18
		sum_19 = 0.5 * count_19
		sum_20 = 0.5 * count_20
		sum_21 = 0.5 * count_21
		sum_22 = 0.5 * count_22
		sum_23 = 0.5 * count_23
		#This sums up the weights and total count of occurences in the paragraph.
		sum_total = sum_1 + sum_2 + sum_3 + sum_4 + sum_5 + sum_6 + sum_7 + sum_8 + sum_9 + sum_10 + sum_11 + sum_12 + sum_13 + sum_14 + sum_15 + sum_16 + sum_17 + sum_18 + sum_19 + sum_20 + sum_22 + sum_23
		count_total = count_1 + count_2 + count_3 + count_4 + count_5 + count_6 + count_7 + count_8 + count_9 + count_10 + count_11 + count_12 + count_13 + count_14 + count_15 + count_16 + count_17 + count_18 +  count_19 + count_20 + count_21 + count_22 + count_23
		#This filters out single-question paragraphs, which add noise to the summary and are usually redundant.
		if len(re.findall('\.', i)) < 1: #Single-question pars won't contain a period. Valuable single-sentence pars will contain a period. 
			average_score = 0
		#This filters out short paragraphs, which are likely to not contain any useful context. 
		#This doesn't apply when a speaker makes a correction.
		#Example of extracted text:
		#Senator Birmingham: The government has provided $2.1 million in the 2017-18 budget to support the authority. 
		#Funding arrangements for 2018-19 onwards will be subject to the usual budget processes.
		#Ms Evans: I might make a minor correction. There is a $1.456 million allocation. 
		#The briefing that we provided to the minister was incorrect.
		#Another example of extracted text:
		#Senator URQUHART: Again, this is to the minister, but if you want to flick it then I am sure you will. Can
		#you confirm that the allocation of money apportioned to the environment portfolio from the $1.1 billion in the
		#rollout of the National Landcare Programme moneys referred to in the budget will stay the same as previously?
		#Will it rise or will it fall between now and 2023?
		#Senator Birmingham: Just for the record, Senator Urquhart, it is $1.1 billion, not $1.1 million.
		elif len(i) <= 55: 
			if re.findall('incorrect', i) or re.findall('correction', i) or re.findall('mistake', i) or re.findall('should have said', i) or re.findall('I meant', i) or re.findall('for the record', i) or re.findall('For the record', i):
				average_score = 1 
			else:
				average_score = 0
		elif re.findall('incorrect', i) or re.findall('correction', i) or re.findall('mistake', i) or re.findall('should have said', i) or re.findall('I meant', i) or re.findall('for the record', i) or re.findall('For the record', i):
				average_score = 1 #Correcting a statement clould require an explaination of more than 55 chars. I'm working on 'sorry' in the context of a correction.
		elif sum_total == 0.0:
			average_score = 0 #Takes care of float division
		#This calculates the mean score of the paragraph. Simple. 
		else: 
			average_score = sum_total/count_total
		#The below threshold ensures that only paragraphs with a high enough score will be included in the summary.
		#Paragraphs that contain high weigthed content will always make it through to the summary for one or more occurences (average score always above threshold).
		#Paragraphs that contain med to low weighted content will need a combination of weights to make it through to the summary.
		#0.6 is not too low where summary includes a lot of noise and is not too high where summary misses a lot of important info.
		if average_score > 0.6:
			summary.append(i)
	return "\n\n".join(summary)
	
def function_2(text):
	paragraphs = text.split('\n\n')
	count_vect = CountVectorizer()
	bow_matrix = count_vect.fit_transform(paragraphs)
	normalized_matrix = TfidfTransformer().fit_transform(bow_matrix)
	similarity_graph = normalized_matrix * normalized_matrix.T #term frequency/inverse doc frequency applied
	similarity_graph.toarray()
	nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
	scores = nx.pagerank(nx_graph) #TextRank applied
	ranked = sorted(((scores[i],s) for i,s in enumerate(paragraphs)), reverse=True) #Sorts all paragraphs from highest to lowest scores
	ten_percent = int(round(10.00/100.00 * len(ranked)))
	ten_percent_high_scores = ranked[0:ten_percent]
	summary = [x[1] for x in ten_percent_high_scores] #Takes top 10%, so the paragraphs with the highest scores (does not disturb the rank order)
	return "\n\n".join(summary)

#Text taken from the user's uploaded PDF or URL, cleaned and formatted.
title = 'ENVIRONMENT AND COMMUNICATIONS LEGISLATION COMMITTEE Estimates (Public) MONDAY, 22 MAY 2017'

#Text taken from the user's uploaded PDF or URL, cleaned and formatted.
#This only includes the text from the body (footnotes, contents, etc can be automatically filtered out).
content = '''
Committee met at 8:58

CHAIR (Senator Reynolds): Welcome. The Senate has referred to the committee the particulars of proposed
expenditure for 2017-18 for the Environment and Energy and the Communications and the Arts portfolios, and
certain other documents. The committee may also examine the annual reports of the departments and agencies
appearing before it. The committee is due to report to the Senate on Tuesday, 20 June 2017 and it has fixed
Friday, 7 July 2017 as the date for the return of answers to questions taken on notice. The committees proceedings
today will begin with general questions of the Department of the Environment and Energy and will then follow
the order as set out in the program.

Under standing order 26, the committee must take all evidence in public session. This includes answers to
questions on notice. Officers and senators are familiar with the rules of the Senate governing estimates hearings,
but if you need assistance the secretariat has copies of those rules. The Senate has resolved that an officer of a
department of the Commonwealth shall not be asked to give opinions on matters of policy and shall be given
reasonable opportunity to refer questions asked of the officer to superior officers or to the minister. This
resolution prohibits only questions asking for opinions on matters of policy and does not preclude questions
asking for explanations of policies or factual questions about when and how policies were adopted.
I particularly draw the attention of witnesses to an order of the Senate of 13 May 2009, specifying the process
by which a claim of public interest immunity should be raised.

The extract read as follows—

Public interest immunity claims

That the Senate—

(a) notes that ministers and officers have continued to refuse to provide information to Senate committees without properly
raising claims of public interest immunity as required by past resolutions of the Senate;

(b) reaffirms the principles of past resolutions of the Senate by this order, to provide ministers and officers with guidance as to
the proper process for raising public interest immunity claims and to consolidate those past resolutions of the Senate;

(c) orders that the following operate as an order of continuing effect:

(1) If:

(a) a Senate committee, or a senator in the course of proceedings of a committee, requests information or a document
from a Commonwealth department or agency; and

(b) an officer of the department or agency to whom the request is directed believes that it may not be in the public
interest to disclose the information or document to the committee, the officer shall state to the committee the ground on which
the officer believes that it may not be in the public interest to disclose the information or document to the committee, and
specify the harm to the public interest that could result from the disclosure of the information or document.

(2) If, after receiving the officer's statement under paragraph (1), the committee or the senator requests the officer to refer
the question of the disclosure of the information or document to a responsible minister, the officer shall refer that question to
the minister.

(3) If a minister, on a reference by an officer under paragraph (2), concludes that it would not be in the public interest to
disclose the information or document to the committee, the minister shall provide to the committee a statement of the ground
for that conclusion, specifying the harm to the public interest that could result from the disclosure of the information or
document.

(4) A minister, in a statement under paragraph (3), shall indicate whether the harm to the public interest that could result
from the disclosure of the information or document to the committee could result only from the publication of the information
or document by the committee, or could result, equally or in part, from the disclosure of the information or document to the
committee as in camera evidence.

(5) If, after considering a statement by a minister provided under paragraph (3), the committee concludes that the statement
does not sufficiently justify the withholding of the information or document from the committee, the committee shall report
the matter to the Senate.

(6) A decision by a committee not to report a matter to the Senate under paragraph (5) does not prevent a senator from
raising the matter in the Senate in accordance with other procedures of the Senate.

(7) A statement that information or a document is not published, or is confidential, or consists of advice to, or internal
deliberations of, government, in the absence of specification of the harm to the public interest that could result from the
disclosure of the information or document, is not a statement that meets the requirements of paragraph (1) or (4).

(8) If a minister concludes that a statement under paragraph (3) should more appropriately be made by the head of an
agency, by reason of the independence of that agency from ministerial direction or control, the minister shall inform the
committee of that conclusion and the reason for that conclusion, and shall refer the matter to the head of the agency, who shall
then be required to provide a statement in accordance with paragraph (3).

(d) requires the Procedure Committee to review the operation of this order and report to the Senate by 20 August 2009.
(13 May 2009 J.1941)

(Extract, Senate Standing Orders)

Witnesses are specifically reminded that a statement that information or a document is confidential or consists of
advice to government is not a statement that meets the requirements of the 2009 order. Instead, witnesses are
required to provide some specific indication of the harm to the public interest that could result from the disclosure
of the information or the document.

I now welcome Senator the Hon. Simon Birmingham, Minister for Education and Training, representing the
Minister for the Environment and Energy, and all portfolio officers. Minister, would you like to make an opening
statement?

Senator Birmingham: No, thank you.

CHAIR: Dr de Brouwer, would you like to make an opening statement?

Dr de Brouwer: Yes, please, Chair. I have some very short comments. On outcome structure, usually we give
a mud map. There have been some changes to the outcome and program structures of the department since the
2016-17 portfolio additional estimates statements. The number of outcomes for the department has been
simplified from five to four. They are broadly environment, climate change, energy and the Antarctic. The major
changes are that land sector initiatives, which were largely the Biodiversity Fund, were previously a separate
program, 1.3, and in sustainable management. They are now included in program 1.1. Program 1.2, environmental
information and research, now includes water matters related to coal seam gas and large coalmining development,
which were previously part of a water program in 4.1. Other water functions have been moved to a new program
1.3 within the environment outcome. To assist you, I have tabled a guide that links the new outcome structure to
key programs and issues. We have tried to fill that out as much as possible.

The second issue is that I understand there is some interest from the committee in asking questions about
Snowy Hydro. Questions on the feasibility study for Snowy 2.0 should be directed to the Australian Renewable 
Energy Agency when they appear. Questions on the government's shareholding in Snowy Hydro Ltd should be
considered under program 4.1 in energy, which is where the shareholder minister's matters are considered.

We have tabled responses to all the 333 questions on notice from the February additional estimates hearing. Of
these questions, around three-quarters were written questions after the hearing. With two estimates hearings left
this year we are currently on track to have around 1,000 questions this calendar year—well above the 10-year
average of about 600 questions a year. As well as providing responses to questions on notice, we have
participated in and responded to an increasing number of parliamentary inquiries. Last year, in calendar year
2016, we had a significant role in 14 inquiries—11 Senate, two House and one joint inquiry. In the first four
months of 2017—this calendar year—the parliament has initiated a further 10 inquiries—six Senate, two House
and two joint inquiries. So that is 24 inquiries. I note that this is resource intensive. We take our obligations to
parliamentary scrutiny very seriously and as such we will endeavour to answer as many questions as possible over
the next two days and we will endeavour to answer all questions on notice from this hearing in the timeframes
requested.

CHAIR: Thank you very much for that succinct but informative opening statement. Thank you for providing
this table. It is very helpful. Given the changes, are there any questions from any senators in relation to where
anything may have moved?

Senator URQUHART: Not yet, but can I just seek clarification. With Snowy Hydro, you said the
shareholding questions are in 4.1, energy. Which ones did you say ARENA will address?

Dr de Brouwer: That is the feasibility study. Snowy is doing a feasibility study and has engaged ARENA as
part of that feasibility study. Those questions should go to ARENA.

CHAIR: Thank you very much for your table and the statistics on the questions on notice and the inquiries. I
encourage you at every estimates, if you would like, to update the committee on the totality. I did not quite catch
it fast enough. There are 24 current inquiries—or this financial year?

Dr de Brouwer: Last calendar year and so far this calendar year we have participated in 24 inquiries.
CHAIR: And we have 10 already this year?

Dr de Brouwer: There are 10 already this year, in the first four months—six Senate, two House and two joint.

CHAIR: On behalf of the committee I acknowledge the very large increase in support for questions on notice
and also for inquiries. Could you pass on to your staff that we are grateful for and we appreciate and understand
the work that it requires—and also for their professionalism in getting them back and providing the quality of
work that you do. Thank you.

Dr de Brouwer: Thank you, Chair.

CHAIR: I now invite general questions.

Senator MOORE: Dr De Brouwer, I have some very general questions around the SDGs. We raised that at
the previous estimates and you indicated that you are one of the core departments involved in the government
response to the SDGs. Can you confirm for me exactly what the input from the department is? How many officers
are involved? Do you have a special unit? We will have some questions later in the research area about oceans,
which comes immediately to mind in your area. But I am trying to get an idea of the investment from the
department in the SDG process.

Dr de Brouwer: I will make some general comments and then I will ask my colleagues. The sustainable
development goals are a priority and are signed on by the government, and therefore we take them seriously. The
work is led by the Department of Foreign Affairs and Trade, but a number of the SDGs are very relevant to the
environment portfolio and we are actively engaged in work around various of them. In terms of the number of
staff, generally the way we work is that if an area—it is more in the responsibilities for delivery of various
elements of the SDGs. Those will lie across many different parts of the department, so there is—
Senator MOORE: Yes, and that is exactly what I am trying to trace in terms of how that approach is going.
Dr de Brouwer: It is very much a coordination approach but focusing also on particular aspects. I will come
back to some of it but I will ask Mr Sullivan to give you more detail.

Mr Sullivan: As you would be aware, Senator, we have the lead responsibility on five of the SDG goals and
we are also a significant contributor to the water sustainable development goal, number six. Also we have input
into other sustainable development goals where lead agencies are elsewhere in the Commonwealth. In terms of
how we are approaching it in the environment department, it is a coordination tasks. I have an international
section inside my division, in which staff are assigned to—the lead responsibility for that is the head of that
section. Other staff support that as need be in terms of the process moving forward. To date, following the 
commitment to the sustainable development goals, we have briefed across the department on what that means,
what the targets are, what the implementation goals look like and how we are travelling with respect to
coordination of that in setting up that regime of coordination. This is something we have done before. While the
sustainable development goals are new, a number of those are not new in terms of that coordination activity from
both the Millennium Development Goals and the Commission on Sustainable Development in terms of national
reporting as well as coordination and input.

The sustainable development goals also contribute in terms of broader policy development, so that is part of it.
That is with the State of the Environment report and a whole range of other factors where we in our division use
the sustainable development goals as a guide, as a contributor to policy implementation and also across the broad
breath of what we are doing in the department. For example, in terms of our reporting structures currently, I also
have a reporting and evaluation component of my division. That area is also looking at the sustainable
development goals to look at how we are tracking with respect to being able to report to that. As you would be
aware, countries have a responsibility to provide voluntary national reports. So we are setting ourselves up in
terms of how we are going to do that, where there are gaps in terms of the targets, and the reporting obligations as
to when that happens.

We are also significant contributors through a working group. I am the representative on the working group,
which is a cross-portfolio working group, and there is also a senior-level interdepartmental committee of which
Mr Thompson is the representative of the department.

Senator MOORE: That is what I was trying to draw out: the various structures that are involved. There is the
IDC, which is the top level—deputy secretary level—and that is coordinated between PM&C and DFAT?

Mr Sullivan: That is correct.

Senator MOORE: But under that there are a series of working parties—that is my understanding. Is your
department involved in a number of those, or is it particularly one group that you are involved in?
Mr Sullivan: There is currently a senior first assistant secretary led working group, for which I am the
representative. That is providing the guidance upwardly to the IDC but also then in terms of information sharing
as to what is happening with respect to consultation, particularly with civil society as well.

Senator MOORE: Do you have a program that is based in your department around consultation on the SDG
or the other term—something 2030?

Mr Thompson: The Addis Ababa action agenda—that one?

Senator MOORE: Yes, I think so. Do you have specific consultation around the role of environment in
SDGs, or is it wider—a part of other things? It is getting a picture of exactly how the SDG component interacts
with the other workload?

Mr Sullivan: I cannot speak for other departments, obviously—

Senator MOORE: No. With these questions we are going to the members.

Mr Sullivan: but, with respect to the environment and sustainable development goals, I am currently the chair
of the Australian Committee for the International Union for the Conservation of Nature—the ACIUCN. That
comprises both non-government organisations and state agencies. We have organised a workshop on the
sustainable development goals, particularly the environment ones but not just limited to the environment related
ones. For example, the health sustainable development goal—

Senator MOORE: When is that?

Mr Sullivan: That is tentatively scheduled for September. That is partly as a way to communicate what is
happening with respect to government processes but also as a way to bring together civil society as well as state
governments and the Commonwealth government, looking forward to how we are tracking and what are some of
the directions coming out of the broader, whole-of-government process.

Senator MOORE: On notice, can I get some information about that particular process in terms of what the
intent is, the planning around it, who is going to go and that kind of thing, as that information becomes available?
I understand that it is evolving at the moment, but can I get that on notice rather than taking this time?

Mr Sullivan: I will try to give you the information now, because the timelines for that may not fit the
questions on notice timeframe. The agreement to that being the next step in terms of consultation came from a
conversation probably two months ago between Australia's regional councillor for the IUCN, Peter Cochrane, and
me. We met with Foreign Affairs and Trade officials to let them know that we were proposing to do this. Foreign
Affairs and Trade and Prime Minister and Cabinet will attend that workshop. The actual timing for the workshop
has not been finally locked down. I have to go through a negotiation phase within the ACIUCN executive. That 
will probably be finalised over the next three to four weeks. That will then be sent to all ACIUCN representatives.
In terms of who will attend that and who will not, I can give you, on notice, the list of who those representatives
are.

Senator MOORE: That would be good. It is not a group with which I am familiar, so it would be useful to
find out what the membership of that group is. Then I will get more information through their processes about
who they are, what they do and all that stuff. This is a joint—

Mr Sullivan: That is one example of the coordination role across the centre of the department. With respect to
the finite goals themselves, we may go into further consultation processes when we have more—particularly
when we move a voluntary national reporting phase.

Senator MOORE: Is this workshop the first workshop or activity that you have coordinated around the
SDGs?

Mr Sullivan: Outside the department, yes.

Senator MOORE: Your first process was internal—internal information about how it works and the roles. So
you have done the education process internally, which I think Mr Thompson mentioned to me the last time I
asked these questions—that that was the process you were beginning at the time. Is that complete now?

Mr Thompson: No, I think it is ongoing. As Mr Sullivan said, the sustainable development goals relate to so
many areas of our business, so we need to reach out to all the bits of the business to help them understand. The
ACIUCN forum that Mr Sullivan has referred to is the first one we have organised. It is not first one we have
participated in. We have been engaged in other fora like the Monash work on sustainable—

Senator MOORE: Can I get a list of the ones you have participated in? I do not think there have been
many—

Mr Thompson: No, there has not been a huge number.

Senator MOORE: but I would like to get a list of the ones that you have—

Mr Thompson: I hope there are more than the one I have mentioned. That is the one I remember.

Senator MOORE: If we can get a list of the ones you have participated in and who attended, that would be
useful. Mr Sullivan, in terms of process, when you do get the kind of statement that is going out to the ACIUCN
members about what it is all about, can I get a copy of the information that goes out?

Mr Sullivan: Certainly—I am happy to provide that.

Senator MOORE: In terms of the process, the domestic elements of the SDG program—there seems to be a
perception that a lot of it is to do with international relationships, and I know that that is true. I have a couple
questions about that as well. But in terms of the domestic work and the domestic reporting responsibilities—Mr
Sullivan, you were talking about the international branch being the part of your area that had the responsibility.
How is the domestic process being handled?

Mr Sullivan: To be clear, the international section forms part of a strategic policy unit inside my division. So
it is not just treated as an international issue; it is being coordinated by that section but in close consultation with
their counterpart domestic team. They sit next to each other.

Senator MOORE: So they are labelled international but it is not just international?

Mr Sullivan: That is right.

Senator MOORE: You said you were getting in place the process for the responsibility of reporting. What
does that engage?

Mr Sullivan: In part it is looking at what data we are already collecting for our—

Senator MOORE: You talked about that at a previous hearing—about not wanting to create new workloads.
You already had a large responsibility for data collection. Is that—

Mr Sullivan: That is right. In part that is looking at State of the Environment reporting and how that can be
translatable into sustainable development goal reporting.

Senator MOORE: In the Australian domestic environment, is the State of the Environment report the core
document?

Mr Thompson: I think it is a foundation document. Because of the nature of that report—it draws heavily on
existing data, indicators and science. What we want to do from that is really to steal the key indicators related to
the goals that are here. The coverage of that report obviously will not go to some of the—the energy goal, for
example, where we have to engage with energy stakeholders as well, and within the department.

Senator MOORE: Is there a reporting process, Mr Sullivan, around this? They are doing the research and
looking at a process, but internally are you getting updates and reports about the state of the SDGs in the
department?

Mr Sullivan: That is still in a formative stage. We are looking at our capability. In part, that will depend on
the timing of the voluntary national reporting process as well. Again, we do not hold that timing. That is—
Senator MOORE: I am trying to find out who does, Mr Sullivan. But it is not yours—I know that.

Mr Thompson: On some of the indicators that we need to put in place for the SDG reporting there is not final
agreement internal to the department, let alone internal to government. Some of that goes, too, to whether and
how we intend to use the SDG reporting as a dashboard for the department domestically as well. Those are deeper
issues which we have still got to work through.

Senator MOORE: Is there any timeframe on that?

Mr Thompson: There is no clear timeframe on that at this stage. I think in part that will be driven by the
timing around the first voluntary national report. We will work towards that.

Senator MOORE: How many times has the high-level working group met, Mr Sullivan?

Mr Sullivan: From memory it has been twice so far. If that is incorrect I will correct it over the period of
estimates.

Senator MOORE: Thank you. Are there minutes of those meetings?

Mr Sullivan: From memory there are records of outcomes and discussion. I am not sure if they are minutes.

That would be a question better directed to Foreign Affairs and Trade and Prime Minister and Cabinet, which are
the lead—

Senator MOORE: I will be directing those questions to them. I was asking you whether there were minutes.
You are not sure whether there are minutes?

Mr Sullivan: I cannot recall.

Senator MOORE: Mr Thompson, how many times has the high-level group met?

Mr Thompson: The IDC—that I can recall, twice. There are meeting outcomes circulated after that.
Senator MOORE: Are they circulated to all the people who came?

Mr Thompson: To the participants.

Senator MOORE: Dr De Brouwer, is there going to be focus on the SDGs in your annual report?
Dr de Brouwer: In future annual reports?

Senator MOORE: Yes.

Dr de Brouwer: The SDGs are a priority of government, and I think that is a natural—

Senator MOORE: So they will be part of the reporting in the annual report?

Dr de Brouwer: It is a natural way to explain and talk about what we are doing. Can I just add to some of the
comments of Mr Thompson around the different forums. Various elements of the SDGs are practically followed
in a range of different meetings we have. For example, the minister had a meeting with the Pratt Foundation
earlier this year around food waste. That is tied to how we deliver the food waste reduction commitments through
the SDGs. Resource efficiency features prominently in product stewardship in the ministerial meeting with the
states—with the environment minister and his state counterparts. But also it is a feature of G20 work.
Senator MOORE: It is all linked in, yes.

Dr de Brouwer: There is a working group on resource efficiency which is looking at how we manage the
efficiency of the whole product life cycle—those issues. There have been meetings in G20 around that. That is
just one set of examples around the SDGs. But they feed through into elements of our work across the board.
Many of them are very practical. So, even if it is not a formal meeting on SDGs, there will be an element that is
carried through from the SDGs in those discussions.

Senator MOORE: Are you working with your counterparts internationally on this process? I know you have
an international relationship on the issues around environment, absolutely. But my understanding from my
discussions with some of the other governments—Germany and the Scandinavian governments—is that they have
integrated SDGs completely into their operational model so that documentation that comes out from their
department, documentation that appears in their international and national statements about domestic policy,
relates to and actually itemises the SDGs and links them to the way they do it. I have not seen that so far in the
publications and the processes that operate within Australian government departments. Is there any discussion of
that degree of integration in your department?

Dr de Brouwer: I will ask my colleagues to respond as well, but I think it comes down to how you deal with
that issue. You are right: as you can see very clearly in German documents, there is a reference back to specific
items in the SDGs.

Senator MOORE: Always—and most of the Scandinavian countries as well.

Dr de Brouwer: It does not mean that if we do not do that that there is not a conceptual reference. The SDGs
are a basic policy frame. It is an Australian government commitment, so it is a basic policy frame. We do not
necessarily link them explicitly or formally but that does not mean that they are not drawn on.

Senator MOORE: Absolutely.

Dr de Brouwer: The way we generally do that reporting is that at the end or in some part of the process we
would say, 'This is how we are going about delivering or achieving the SDGs in Australia'. So there is a separate
process. I just make the point that countries can do it in different ways. They can either be very formal and
explicit around how they draw on it, or it is an implicit but still very firm drawing on that basis.

Mr Thompson: I think that is right. Australia typically has taken the less formal approach that Dr De Brouwer
was talking about. I can give an example from the Convention on Biological Diversity, where we have the Aichi
targets. Those targets have been hugely influential in policy setting and within the department, but when we go
and talk to the Australian public and when the government talks to the Australian public about the national
reserve system or some of those other elements we do not tend to use that as the point of reference. It does not
mean that it is not significant.

Senator MOORE: So it is permeated in the culture of the department?

Mr Thompson: In that case I think it definitely is. Is sustainable development goals permeated fully yet? No,
I do not think so. But it is not a very old agreement of government at this stage. That is an evolving conversation
that we are having within the department.

Dr de Brouwer: I will go back to a comment that Mr Sullivan raised at the start. When the SDGs were
agreed, we then had a formal process internally, through division heads, of talking through what Australia's
commitments were—going through all the SDGs and how they relate to our work. So we do formally discuss
those and there is a proper discussion around it, but that does not necessarily mean there is tracking at every point
in time.

Mr Thompson: They are referenced in the State of the Environment report and they are referenced in our
corporate plan.

Senator MOORE: I asked about the department's staffing allocation but, Mr Sullivan, you said that it is
combined with other duties, so it would very difficult to indicate what the investment of the department is in
terms of resourcing for this issue.

Mr Sullivan: It is on an as-needs basis. It is resourced and, as I said, led from the international section but—
Senator MOORE: But there is no dedicated resource?

Mr Sullivan: There is a dedicated task but, in terms of staffing—for purposes of building corporate
knowledge inside the organisation there are lead staff inside that section. But at times of peak workloads those
staff will contribute as required.

CHAIR: I would like to pick up this issue of sustainable development goals. Secretary, I went to Indaba this
year to represent the government. One of the things I reported back on was the amount of work that Australian
mining companies are doing at 600 sites across Africa. There were a lot of really fantastic examples of working
with local communities on environmental programs—agriculture, so not within your portfolio, but they are clearly
assisting local communities. While there is not a model for this, I think there is a case to put a model, because
there is a lot of helping to achieve sustainable development goals in these local communities—I looked at up to
about 10 of them—led out of the mining sector on environmental and community issues. Are you aware of what
they are doing over there? I know you would be aware of it, but have you drawn any link to assisting them to
achieve sustainable development goals?

Dr de Brouwer: In broad terms, yes, we are aware of it. The point you make is that the responsibility for
sustainable development is one that is shared across the community. Government plays its role but so does
business and so do local communities, NGOs and others. So it is in that sense a shared responsibility. How we
account for that, how we talk about that, I think is a good question to raise with us. My sense is that we do not talk
enough about what the role of others is in the process. 

CHAIR: It struck me because Austrade and DFAT are aware of what they are doing but it is not specifically
their responsibility. They look at it more as a mining, energy or environmental issue. Would you mind taking on
notice for me to see if there is a way that it can be done cross-portfolio? There is a lot of good work and if we
could capture that model—what has been done in the environment, in your portfolio, and in others—it would be a
great template for others globally but also, to pick up on the issues Senator Moore raised, bring some of those
lessons home as well.

Dr de Brouwer: We will do that.

Mr Sullivan: Chair, you have made a really good point in terms of demonstrating and showcasing. Part of it is
quantitative reporting under targets, but also traditionally this is an opportunity to showcase things that are
happening internationally through aid programs or through cooperative mechanisms and public-private
partnerships and also domestically. For example, in the mid-1990s Australia made a presentation based on similar
sustainable development goal reporting on the creation of Landcare in Australia—showcasing that as an
Australian example. What resulted was an international Landcare movement that came directly out of that
showcasing. One of the things around sustainable development goals is that it presents an opportunity for
government, civil society, public-private partnerships and industry to showcase the good things that are
happening. It is a really important part of it.

CHAIR: Picking up your point, Mr Sullivan, a lot of these companies are doing pre-rehabilitation
rehabilitation of the land, a bit like landcare—environmental rehabilitation using new agricultural techniques as
well, so it is a combination of things. There are some good messages out there.

Senator MOORE: I have some general questions around the State of the Environment report. In terms of the
recent one, which I understand was released earlier this year, what was the process? Is there a media process
around the release of the State of the Environment report? As you said earlier, it is the foundation document. I
really like that term—it kind of sums it up. What is the program around when it comes out? How is it promoted?

Mr Thompson: The State of the Environment report is the most comprehensive assessment of Australia's
environment across a number of domains. It is a report which is required under the EPBC Act to be tabled by the
Australian government in the parliament every five years. The centrepiece action in terms of promoting it is really
the minister's decision to table the State of the Environment report. Importantly this year it takes the form
primarily of a digital platform rather than a physical paper report. There is a short overview which is available in
paper form but it is the digital platform which is the tool that we are directing people to. Around the time when
the minister tabled the report he wrote a piece—for the Guardian, I think—that was published at that time. The
tool was made available on our website and also has its own dedicated website, as you would expect for a
platform of that kind.

We have also been working and had done some prior spadework on with a number of organisations, including
the Global Compact for business, to promote the State of the Environment platform. The lead author, Dr Bill
Jackson, participated in a webinar relating to the State of the Environment and informing members of that
compact of the main findings of the report. We are also rolling out a range of other communication tools and
products, including through the Australian Environmental Grantmakers Network, which is the philanthropy
sector; the education sector, through teachers who teach environment and science; and a number of other quarters.
We will provide briefings to senior officials of the jurisdictions of state and territory governments in the future as
well.

Senator MOORE: Can we get a report on the program around that? It is a wide-ranging program you have set
out, and this is into the future. It has come out in a different format this time, so there is a kind of threshold
difference. Can we get some information from you in detail as it comes through rather than taking up too much
time talking about it?

Mr Thompson: Yes, we are happy to do that.

Senator MOORE: Has there been a response to the changed format? Have you had immediate feedback
about it—because it is significant different?

Mr Thompson: It is significantly different in terms of the platform. What is pleasingly not different is that we
are using a similar format to the 2011 report, so you are able for the first time in the history of State of the
Environment reporting from when the EPBC Act came in to track more directly—not completely, but more
directly—from the 2011 report to the 2016 report. We have had very positive feedback about that. We have also
had very positive feedback about the digital platform and the way in which people are able to work with it. There
are over 300 maps and charts in that report. That material can also be taken and pulled apart and assembled in
ways people want to use it to answer particular questions that they have, whether they are policy questions or land
use or natural resource management questions. All of the data sources that underpin the findings of the State of
the Environment report are accessible in the layer beneath the digital platform. In terms of the transparency of the
reporting and the independence and rigour around the reporting there has been very strong feedback, and we are
building on a very good product.

Senator MOORE: In terms of the process—these are similar questions to those I asked about the sustainable
development goals—are there staff permanently allocated to the issues around the State of the Environment
report? At the time of the creation, of course—but in that five-year in-between period is there ongoing work
within the department on reviewing it, evaluating it and working on how it should best work? Is there a team that
does that?

Mr Thompson: Clearly, as you say, there is a largish team under a director who is responsible for preparing
the State of the Environment report and supporting the independent lead authors. That is a sort of surge activity
over 2½ or three years. That surge has now come off but we still have a State of the Environment team under a
director. Their role now is really to undertake some of the communications activities that we have identified to
promote the digital platform within the department, across the Commonwealth and more broadly, and also to
continue to explore how we can keep the State of the Environment as a more current report using the digital
platform. They are working now on that range of issues. I could not tell you how big that team is. It would be in
the order of four or five people.

Senator MOORE: Can you take that on notice?

Mr Thompson: I am happy to take that on notice.

Senator MOORE: Is that the same strategy as for the last one, or is that an evolving strategy of having a
team?

Mr Thompson: It is not dissimilar. The difference this time is that we have probably got a couple more
dedicated people on State of the Environment in the post-reporting period than we did last time, if I recall
correctly. That team will also be assisting with other parts of the department's agenda in relation to essential
environmental indicators and those sorts of things, which also support State of the Environment reporting.

Senator MOORE: So it is kind of dynamic usage of the people with the expertise?

Mr Thompson: That is right.

Senator MOORE: The recent report had some fairly serious comments about lack of leadership, policy and
follow-up on policy action. It was through some of the statements that were made about the progress between
2011 and the latest one. I know the department cannot comment on that. Minister, can the government comment
on some of those findings that were in that report? They are clearly around lack of leadership. The State of the
Environment report questioned the degree of leadership around environmental issues.

Mr Thompson: Senator—just to clarify—I think the authors did reach a conclusion that there was a need for a
national overarching framework.

Senator MOORE: They did, yes.

Mr Thompson: They also clearly identified that the importance of Australia's natural capital and the
environment and the challenges we face in managing a continent and the marine resources we are responsible
for—the stewards for—is bigger than the national government or any state government alone can manage.
Senator MOORE: And a responsibility.

Mr Thompson: And a responsibility—not only for those governments but for other partners, including the
business community, NGOs and the wider community. In that context they did point to the need for the various
partners to collect around some of the clear priorities identified in State of the Environment 2016 and the
challenges there and to pursue action together to meet those. Coming back briefly to the latter part of your earlier
question about the department's response to those things—as I said, we have a small SoE team ongoing, but the
State of the Environment is used across the department to help prioritise our spending programs, to help inform
our advice to government on policy, to identify emerging risks in relation to regulation under the Environment
Protection and Biodiversity Conservation Act and, importantly, also to inform the natural resource management
that we undertake ourselves, whether that is in Antarctica or in our parks estate under the Director of National
Parks and the Commonwealth Environmental Water Holder. That information informs us, forms part of our
ongoing work and forms part of the government's response to that report.

Senator MOORE: And leadership? They laid down a challenge in terms of the serious nature of
environmental need and the need for everyone to play a role. Certainly from my reading of the overall front page
document—I have not gone through all the digital underlay of this—they said that there needed to be key
leadership in this area.

Mr Thompson: I think there are areas where the Australian government is providing leadership: through the
environment ministers meeting decision of November last year to undertake work and develop a strategy on a
common national approach to environmental economic accounting; through the work we are doing with the states
and territories and with the business sector and other NGO partners on threatened species and pursuing innovative
approaches to protection and support of threatened species; and through a number of other mechanisms being
pursued with the states—for example, the National Clean Air Agreement and those sorts of things. I think the
Australian government is providing leadership in respect of those particular issues.

Senator MOORE: I am taking down those notes—it was the common approach element, various aspects
around threatened species, and clean air. Is that—

Mr Thompson: The National Clean Air Agreement.

Senator MOORE: From the department's perspective, are they the key elements of the government's leading
of environmental change?

Mr Thompson: They are key examples. I do not think they are the only examples.

Senator MOORE: What would you add, Mr Thompson—or Minister—as to where the government is leading
environmental action? As pointed out by the SoE, there needs to be leadership in policy implementation and also
policy follow-up. That follow-up issue was clear as well. I know we cannot take all of Senate estimates to say
what the government's leadership is but, just in terms of a snapshot, what are the leadership areas?

Mr Thompson: I would add two examples. One would be the Murray-Darling Basin and the leadership that
we have provided across a number of governments on sustainable diversion limits, the work of the
Commonwealth Environmental Water Holder. And there is the leadership that we are providing across the field of
climate change. I will not go into detail on that but it is an area where the government is providing leadership.
Senator MOORE: It was highlighted in the SoE as an area that needed—climate change was listed there.

Mr Thompson: The SoE did have a chapter on climate change and atmosphere. The other area would be
around Commonwealth marine reserves and the protection provided there, which the State of the Environment
clearly pointed to as an improvement from 2011. The declaration of those reserves and the management plans for
those reserves are currently being considered by the department.

Dr de Brouwer: I would also add the focus on making sure that instruments are directed to environmental
outcomes, and also to economic and social outcomes—that the nature of regulation and programs is for a
particular purpose: to conserve and protect the environment while achieving a broader set of goals. That outcomes
focus is now much clearer in the implementation and application of environmental regulation. That involves
explicit focus on outcomes, co-design with proponents and risk-based approaches, as well, to managing
environmental risk—where those risks are greater, stronger risk management and tighter compliance or tighter
condition-setting. With the nature of programs there is greater focus on procurement—so getting environmental
outcomes at a local level, working with groups, rather than grant programs. It is about trying to get the
instruments more focused on the environmental outcomes. That is a pretty fundamental feature and it ties in well
with the State of the Environment report.

Senator MOORE: Would that process you have outlined with regulation and focus on outcomes include
follow-up in terms of—once you make a decision and put a regulation in, how you follow up on that?

Dr de Brouwer: Yes. Compliance is very important. Efficient and effective compliance means taking riskbased
approaches. That is now formally and expressly—

Senator MOORE: It sounds like a T-shirt.

Dr de Brouwer: Sorry. It has been a very big change in the department. It has meant that you identify the
risks and then you spread your resources and focus on where the greatest risks are. That approach has been
applied over the past couple of years more intensely. Again, that comes back to ensuring those environmental
outcomes. It is not meant to be glib management-speak.

Senator MOORE: No, but you know what I mean. That kind of statement is like a challenge—it is a little,
easy statement but it is about making that work.

Senator Birmingham: Turning it into reality.

Senator MOORE: Yes.

Senator Birmingham: If you can get T-shirt sales for undertaking risk-based assessments—

Senator MOORE: They would have to be pure cotton and made in Australia—

Senator Birmingham: and approaches to take off around the country, then regulatory experts will be thrilled
at the enthusiasm for their policies.

Senator MOORE: Would that it would happen, Minister. One of the things the SoE said is that there was a
lack of an overarching national policy on the environment. That follows on from some of the questions that I have
had about the SDGs and the SoE. Minister, I know that you are representing the minister here and it is not your
area, but what is the response from the government to the statement that there is a lack of an overarching national
policy? This is what we were talking about—that there is something that includes all levels and gives a clear
vision.

Senator Birmingham: Across all of the different facets of environmental regulation and stewardship there
have been some great strides forward in terms of the collaborative approaches between the Commonwealth and
other stakeholders, particularly the states and territories. Mr Thompson cited some of the work around the
Murray-Darling Basin before. That is one area. I expect that in the next little while we will turn to talking about
the Great Barrier Reef. Despite some of the very serious challenges there, I think the Reef 2050 plan and the work
there of far more integrated collaboration between the Queensland government and the Commonwealth
government that perhaps has historically never been the case in terms of reef management and stewardship is an
important example of us finding ways for the different parts of government to provide a clear national element of
leadership in a coordinated manner. Overall there are several key frameworks that bring together environmental
policy.

The EPBC Act remains a coordinated approach at a federal level. Again, there have been efforts, which have
made some progress, to better align EPBC processes with those of some of the states and territories over recent
years. The Water Act provides a similar framework particularly for the Murray-Darling. And of course our
commitments under the Paris Agreement provide the framework that we use in trying to get the states and
territories to take a more coordinated approach to emissions mitigation and reduction policies and that we apply
across a range of other areas of government, particularly through the energy space. The work that the Finkel
review is doing will, we hope, bring the states and territories to something that is more coordinated in that space
too.

Senator MOORE: Dr Finkel is coming in tomorrow, isn't he?

Dr de Brouwer: I do not think so.

Senator Birmingham: He might be appearing at Industry.

Senator MOORE: But in terms of the work that he is doing—

Senator Birmingham: I am sure that at Industry he will be able to cover—

Senator MOORE: He is not appearing for Environment?

Senator Birmingham: He is not appearing here.

Dr de Brouwer: I think that in Industry he will be appearing as the Chief Scientist rather than—

Senator MOORE: On the work he was doing in the process—okay. Coordination is critical. I have taken
down—which will read in the Hansard better than my own notes, which are a bit jumbled. The marine reserves
area is one that we are really interested in in terms of that, and it was particularly mentioned in the SoE in terms
of the importance of that area. Has the work in that space been put on hold by the current government?

Mr Thompson: No. The government has asked for work to be done to develop new management plans for the
reserve zones within the reserve boundaries. That work is being led by the Director of National Parks, who will be
appearing a bit later this morning.

Senator MOORE: So there is no deferral of the work in that space?

Dr de Brouwer: Ms Barnes will be here—I think scheduled for 12:15.

Senator MOORE: Okay, we can ask then. Will the government or the department be making responses to the
statements in the SoE?

Mr Thompson: At this stage there is no intention for the government to make a single coordinated response to
the SoE. But, as we have tried to map out in the answers we have given here collectively, the response really
comes through the ongoing work of the department in the various areas and domains that the SoE covers.

Senator MOORE: The list of priorities you gave that the government is taking on board in areas around the
clean air process, the Murray-Darling and climate change—they are all mentioned there. You said the education
programs are going to allow more work in these spaces, get more information and encourage the discussion
around them. Will they be processes where there will be statements from the department about what the ongoing
action is?

Mr Thompson: Not comprehensively framed around the State of the Environment or using SoE as our
ongoing dashboard, if you like—that is not the intention at this stage. But each of those processes that we referred
to, or each of those domains that we referred to, whether it is climate change or the Great Barrier Reef, as the
Minister pointed to, which is a significant bilateral engagement with Queensland—

Senator MOORE: And an ongoing process in terms of—

Mr Thompson: And ongoing—and others that we did not refer to like the national Heritage Strategy and
those sorts of things. They each continue to engage with stakeholders and provide updates. There is a review
which has just been undertaken on Australia's Biodiversity Conservation Strategy, for example, which is a joint
piece of work that we have with the states and territories. The state and territory ministers and Minister
Frydenberg agreed that there would be a refresh of that strategy, and that work is being undertaken now. That is
just one example of how these things will roll on. We also have regular updates to government, within
government and then externally on progress being made against Australia's Antarctic strategy and 20-year action
plan in terms of how we are meeting our stewardship obligations in the Antarctic. Those are just a couple of
examples.

Senator MOORE: They are reported to government and then put on the website—

Mr Thompson: In various forms—that is right.

Senator MOORE: so that people who are interested in the space can then come back and do more?

Mr Thompson: That is right.

Senator MOORE: When a big document such as the SoE goes out, the people who are interested in this
processes do follow it very closely. I am not quite sure how you translate that into the wider community taking an
interest, which is the challenge for all of us. Has any consideration been given by the department or by
government to doing a response, or has it just been a decision to take another course?

Mr Thompson: Within the department certainly when we were framing the State of the Environment report
we did give consideration to the advice to government about whether there should be a single comprehensive
response. The advice was that that was probably not warranted. In part it goes to the issue that you have raised: it
is such a large and comprehensive report, covering so many domains of Australia's environment, that trying to
capture a response to that at a point in time would not necessarily be the best use of our resources. We would be
diverting people from the work they are currently doing to try to do that. Instead, what we are pursuing are the
various agendas that feed the material issue, which is improving the state of Australia's environment.

Senator MOORE: Is the department considering moving to what is called a dynamic State of the
Environment report that updates regularly online when new significant data becomes available and at the same
time producing static SoE reports at periodic intervals that pull together and analyse all of the data? It seems to
me that some of your answers have lead down the track that that is what you are doing, but I just want to get that
clear.

Mr Thompson: It is the holy grail. Of course we would like to do that.

Senator MOORE: Absolutely the holy grail.

Mr Thompson: I would make two distinctions—and you have framed them in your question. One is updating
the data that underpins State of the Environment reporting. That is something we are actively looking at: how we
keep the data within the State of the Environment more live, more contemporary. I do not think we would move
to a situation where we are almost in real time updating the qualitative assessments that lead authors do on the
state of Australia's environment in those domains. That would be too difficult. I am making a distinction between
that and the data that underpins State of the Environment reporting and keeping that more contemporary, which is
something we are looking at. The other value of SoE is that it represents a scientific consensus, if you like, based
on a lot of peer review and a lot of consideration and qualitative judgement—not necessarily quantitative—based
on the data about the state of Australia's environment. Keeping that contemporary is very difficult. I do not think
we would be looking at a situation where that part of the SoE remains live over the five years. Having said that,
there are other parts of the department which undertake reporting which does not seek to keep that live but
intersperses the five-year period with other reports. Examples include the work by the Bureau of Meteorology and
CSIRO on the state of climate, and of course the GBR outlook report, which is a significant piece of work as
well—on which State of the Environment is currently modelled in some ways.

Senator MOORE: There is a commitment from the government to keep those datasets up to date, because
they are your measuring tools?

Mr Thompson: There is an aspiration from the department to keep them up to date, and we are exploring how
we might do that. We are talking in the State of the Environment context but there is also work underway in
relation to essential environmental indicators by the department and how to get common agreement across the
nation on which key environmental indicators we want to track, and then separately as well the environmental
accounting work that I referred to earlier.

Senator MOORE: And that also links back to those original questions about the reporting around the SDGs.
If you can get agreement around what datasets are used by whom and for what, and then make sure they are up to
date, that would be a step towards it. And if they can link through to the SDG agenda, that provides the basis for
your reporting in that space.

Mr Thompson: That is right.

Senator MOORE: But for the qualitative processes you are going to rely on issues-based pieces of work that
can then be linked back to the SoE on key issues such as marine reserves?

Mr Thompson: Yes.

Senator ROBERTS: Cost of living is a very significant issue for people in Queensland and across Australia.

In Queensland we are now heading for a doubling of cost of energy for consumers and industry compared with
recent years. In fact one supermarket chain, a small chain, had a 20 per cent increase from last year to this year in
its energy bill. That is significant. We in this room are all paid by taxpayers. The Australian people would like to
know what you are paid, Dr Brouwer, on an annual basis including all entitlements.

Dr de Brouwer: I will have to take that on notice. I do not have it here. My salary is on the public record
though. But I will come back to you.

Senator ROBERTS: Thank you. Can you also provide the position titles of all staff who receive salaries all
inclusive of over $250,000 a year?

Dr de Brouwer: I will take that one on notice. There is a discussion around how we report executive salaries,
and I want to be as transparent as possible around that, so let me take that one on notice.

Senator ROBERTS: Thank you. Energy is increasing in price. For someone on $40,000 a year income, the
energy prices they are facing at the moment are like driving over a cliff. On our salaries, and on your salary as a
department head, that just becomes a pebble to some people. It is very significant then—I would like to know
what the renewable energy target is running at the moment. Is it still 23 per cent?

Dr de Brouwer: It is a megawatt hour target. It is 33,000 megawatt hours a year by 2020. That then becomes
equivalent to, based on the projected demand for 2020, 23½ per cent. We will confirm exactly what that number
is, but it is formally a megawatt hour target.

Senator ROBERTS: And what is it right now? Where are we at?

Dr de Brouwer: I think we are at around 17 per cent. I will ask Ms Evans to—

Ms Evans: The team who are able to answer the detailed questions on the renewable energy target are the
Clean Energy Regulator. They are on later in the program. I do not have the detail of exactly where we are at
now.

Senator ROBERTS: What is the basis of the renewables target? On what advice has that target been set?
Ms Evans: There was an extensive review of the renewable energy target done in 2014, and the target was set
through bipartisan agreement at 33,000 gigawatt hours. There was a range of modelling and other work done, and
that report is in the public domain. We can arrange for a copy to be sent to you if you would like.
Senator ROBERTS: On whose advice was that bipartisan decision was made—what body?

Dr de Brouwer: It was a government decision.

Senator ROBERTS: And what underpinned the government's decision?

Dr de Brouwer: There was a major review by Mr Warburton—the Warburton review—on the renewable
energy target. I would have to go back to that document, but the government, based on that report and in
negotiations in the parliament, settled on 33,000 gigawatt hours. Sorry—earlier I should have said gigawatt hour,
not megawatt hour.

Senator Birmingham: A minor difference. Essentially, Senator Roberts, this is testing my memory but I think
there was a degree of concern at the time that the scale of the renewable energy target as it had been set was
excessive relative to where energy demand had been heading. There were concerns about whether the targets
could be met and there were elements of concern about the way it was working at the time, and hence the review
was undertaken. The determination of the government at the time—which, if my recollection is correct, ultimately
received bipartisan support—became to adjust the gigawatt hour target.

Senator ROBERTS: Is the burden of proof for environment and energy policies on the department, others
within the government, or others outside the government?

Dr de Brouwer: The department provides advice to the government. It does that through the cabinet process.
That is normally cabinet-in-confidence—the way we engage with the ministers in the cabinet. But the government
takes into account a wide range of views. Frankly, it takes account of every view across society in making its
decision. That is then a decision of government.

Senator ROBERTS: We now have policies in many states and federally for reducing the output of carbon
dioxide. On whose advice was that based originally?

Dr de Brouwer: I think that there has been a long, steady stream of advice that has gone to government
through various government bodies and task forces over a number of decades as well as public discussion around
those matters.

Senator ROBERTS: Has anyone ever challenged that basis that you are aware of?

Dr de Brouwer: I think there is a range of views on these matters, including your own views. That has been
part of the public debate.

Senator ROBERTS: I guess what I am coming to, Dr de Brouwer, is that, from what I can see and from what
I believe, good policy is based on sound evidence—empirical evidence: measured data and physical observations.
I have been chasing this down, as you would be aware. I have not found any that shows that human production of
carbon dioxide is affecting the climate and must be curtailed. So I would like to know where that comes from.
That is the first question. Secondly, I would like to know: has anyone challenged that and asked for that empirical
evidence?

Dr de Brouwer: The government—the department and then also the government receive their advice around
science from a range of government scientific bodies and from non-government scientific bodies, especially those
based in universities. As you know, we have had this discussion before, including in answers to your questions on
notice. We do draw very heavily on the Bureau of Meteorology and CSIRO but also the Chief Scientist, the
Academy of Science, universities and other international processes. The overwhelming assessment from those
scientists is that carbon dioxide or greenhouse gases are a cause of global warming and predominantly that
greenhouse gas increase is related to human activity. They also explain that that is related to—that can be related
back through the chemistry of carbon dioxide and other greenhouse gases to the burning of fossil fuels. That is
what the scientists say, Senator, and we draw on their advice. Those assessments are peer reviewed. Not
everyone, as you say yourself, agrees with them. But they go through an intensive peer review process that is
open and public. So, frankly, the best we can do is to draw on the advice of the experts in an open and transparent
system. That is the basis on which we then form our advice. That is not saying that we are not listening to others,
but listening does not mean, again, that there is an agreement.

Senator ROBERTS: Sure. Are you aware of anyone doing any due diligence on that advice in the last 10
years or so?

Dr de Brouwer: I think the nature of the due diligence comes through in the IPCC processes and the scientific
review and peer review. That is an open process, so that draws on a wide—globally, on scientific evaluation and
review. That system and that process is open for that.

Senator ROBERTS: On what basis do you say that the IPCC is an open process?

Dr de Brouwer: It draws on a broad range of work of scientists, and there is a peer review among scientists
and others. That is subject to public scrutiny. It is transparent. Transparency is the greatest defence, because it has
public scrutiny.

Senator ROBERTS: So you believe the IPCC is transparent?

Dr de Brouwer: The work is made transparent, and the work is publicly available. Again, transparency is our
greatest defence.

Senator ROBERTS: I would agree. If it existed, transparency would be our greatest defence. Is the standard
of proof then, for at least major policies, beyond a reasonable doubt—high—on the balance of probabilities—
medium—or based on the precautionary principle, which would be very low? What is the standard of proof?

Dr de Brouwer: It is a very general question, and it is very hard to—

Senator ROBERTS: It is.

Dr de Brouwer: Normally we would apply that in specifics.

Senator ROBERTS: I am talking not about climate but about energy.

Dr de Brouwer: Energy?

Senator ROBERTS: Yes. It is a major policy that—

Dr de Brouwer: It is, and the way that the government—

Senator ROBERTS: is disrupting Australia.

Dr de Brouwer: It is very hard to answer that, Senator. The probabilistic judgements that are made are really
then—it is less that and more what the government's objectives are around dealing with energy matters. It is very
clear in the budget papers that the government is looking for three features of the energy system. It is looking for
a reliable, affordable and sustainable energy system now and over time. That is its focus. Then how it makes those
evaluations or the policies to achieve it is then a matter for consideration within the government. The probabilistic
basis for that really depends on the particular elements and the risks that the government is willing to take.

Senator ROBERTS: So it is not possible to say whether, in making major policy decisions, you rely on
beyond a reasonable doubt, on the balance of probabilities or the precautionary principle?

Dr de Brouwer: At various times I think governments rely on all of those elements to different degrees.

Mr Heferen: Can I just add to Dr de Brouwer's answer. The other issue to bear in mind is that a lot of energy
policy is not determined at the federal government level. It is a federal decision—that is, the states and territories,
I guess, have the key responsibility for decision-making for the energy system and, as you are probably aware,
what all of the states and territories have done is come together in a collaborative way mostly—not always but
mostly—to try to make decisions in the national interest. At the moment, our national energy market, the NEM, is
really just the east coast and South Australia. Western Australia has its own system and the Northern Territory has
its own system. But, in the NEM, as you probably know it is referred to, the Commonwealth environment and
energy minister chairs that, but each state and jurisdiction or each state and territory involved—so each state and
the ACT—essentially has an equal say. Of course, as you have also alluded to, states will sometimes act
unilaterally and make decisions that they consider in their own best interests. There is a constant debate.

Should all decisions be made to try to maximise or optimise aggregate welfare—welfare in an economic
sense—across the NEM? When is it the case that a particular jurisdiction feels it needs to step out of that and
make its own decision? Whether that is on beyond reasonable doubt, the balance of probabilities or the
precautionary principle, as Dr de Brouwer said, will vary on a case-by-case situation. I think it would be fair to
say that rarely would there be a breakout where one would act unilaterally and you would go back and say, 'Well,
actually, it is beyond reasonable doubt'. Most things would then be on the balance of probabilities.

Senator ROBERTS: My experience in federal parliament shows that there is very low accountability in this
institution. So I would put it to you, then, Mr Heferen, that the national energy policy could at times be likely to
be hijacked politically because various states—that is what it seems to be driven by. That is not a question—that
is an opinion—

CHAIR: Senator Roberts, can you confine it to questions. You are straying into hypothetical conversation at
this point.

Senator ROBERTS: Sure. It is just that I am feeling a little frustrated that we cannot seem to pin that answer
down. But it is a broad question. Does proof for at least major policies include rigorous and independent costbenefit
analysis which best captures the triple bottom line, especially the economic aspect?

Dr de Brouwer: Yes, I think regulatory impact statements, for example, are meant to be a form of cost benefit
analysis that informs major decisions and decision-making by government. That is a normal part of presenting the
arguments for and against. Cost-benefit analysis is a standard form of analysis that underpins advice.
Senator ROBERTS: Are we able to get copies of cost-benefit analysis from the department?

Dr de Brouwer: Yes, but what is the specific—

Senator ROBERTS: In regard to the formulation of the 23 per cent target.

Dr de Brouwer: I would have to go back to the Warburton Review. That went through the pros and cons.

Senator ROBERTS: And costed?

Dr de Brouwer: I cannot recall. I would have to take that on notice. But we can come back to what RIS
process was done for that.

Senator ROBERTS: I would like to know the economic cost benefit in particular, not just the triple bottom
line but especially the economic cost benefit analysis.

Dr de Brouwer: Okay, we will take that on notice and we will come back later.

Senator ROBERTS: Has any work been done on evaluating both the history and the theory that shows that
natural resources and environmental management systems based on voluntary and competitive private property
rights are far more efficient and effective and thus more sustainable than those based on involuntary or
monopolistic command and control ones? I am looking at freedom versus control—in other words, control
policies versus those free-market policies on environment.

Dr de Brouwer: I think that is such a very general question that—

Senator ROBERTS: Has anyone evaluated whether or not we should have an environment policy based on
freedom of enterprise or on command and control through government?

Dr de Brouwer: I think the way we typically go through that is again around regulation—whether the burden
of the regulation is a net gain or there is a net social or a net economic gain from that regulation. I think we would
generally—the regulatory impact statements and regulatory focus or deregulation focus of the government has
required an assessment of the value of that regulation—is it valuable or needed or not. So there is not a hyper or
an overarching philosophical discussion. It really comes down to the application, in particular circumstances, of
whether a regulation is required or not.

Senator ROBERTS: So there has been no study as to whether or not a freer approach or a regulatory
approach is more effective? It just comes down to each regulation?

Dr de Brouwer: I think there are approaches to regulation that would encompass features of what is the
purpose of regulation and the purpose of individual choice or freedom. We do go back, as I said before to Senator
Moore, to the focus on outcomes. We try to get a greater focus on outcomes, which are environmental but also
economic and social outcomes, that are co-designed with proponents—and this is in our regulatory system—and
risk-based approaches. So how you go about regulation really does matter. Regulation is something you—just
doing it to people can be counterproductive. It is about how you design a regulation that is going to achieve the
outcomes that you aim for—and that is usually in legislation, which is what we take to be the public will—and
also that is properly conditioned and has the compliance around those conditions.

Senator ROBERTS: So do control policies then face the burden of proof and face a high—beyond a
reasonable doubt—or a medium—on the balance of probabilities—standard of proof rather than a low one—the
precautionary principle?

Mr Knudson: Just to pick up on the Secretary's comments about how we use regulation to take a look at a
range of options, there are some really good examples, and we can talk about this later on in the program, with
respect to product stewardship and how we deal with different issues, where the regulatory impact statement will
look at whether we are better to go with a voluntary agreement where we are trying to achieve some objectives
working with industry in a collaborative way et cetera, which is more on the end of personal freedom, as you
would have put it, all the way to a steadfast Commonwealth or state regulation. It canvasses those issues and takes
a look at the cost benefit associated with each of those. We have done this for a number of different products
where there is science which indicates that there is a risk to the environment, and that is what underpins that work
to take a look at those different options. I just wanted to flag that we are happy to talk about that in more depth
when the officers are here and we can go through some specific examples to illustrate how we have done that.

Senator ROBERTS: Thank you. Would you agree, Mr Knudson, that more prosperous societies can afford to
spend more on protecting the environment and caring for the environment?

Mr Knudson: It is a hypothetical question.

Senator ROBERTS: Cheap energy is fundamental to protecting the environment: has any research been done
within your department on the veracity of that statement? In other words, will high energy costs hurt the
environment directly or indirectly?

Dr de Brouwer: I am not aware of that. We will take that on notice.

Senator Birmingham: I think the government has made clear, Senator Roberts, that our ambition is
absolutely to keep energy as affordable and reliable as possible whilst simultaneously dealing with environmental
issues associated with energy. Essentially, there are three pillars behind the energy policy we seek to pursue:
environmental responsibility, affordability and reliability. We see all three as being integral to sensible policy
solutions, and that is a core part of what the Finkel review is looking at too.

Senator URQUHART: I just have a couple of questions on the State of the Environment report before I go on
to some other areas. I guess I want to sum up Senator Moore's questioning—that is, what is new in the State of the
Environment report since this government came into power?

Mr Thompson: Can I just clarify. Are you asking what is new about the nature of the report or about the
findings?

Senator URQUHART: I guess it is probably the nature of the report. We know that marine reserves have
been put on hold. Boundaries have been declared, but then what happens in them. So the management plans have
not been put in place, as I understand, and that is currently being reviewed.

Mr Thompson: That is right.

Senator URQUHART: You raised the issue before about marine reserves. What has changed in the ocean,
basically, as there is no management plan? What are the things that have changed? What is new?

Mr Thompson: I think in general terms—I will start and try to list some of the issues raised. Clearly, what the
State of the Environment found was that the marine and Antarctic environments are generally in good condition,
that natural and cultural heritage areas are likewise and likewise the air quality of our cities. They are generally in
good condition. In terms of the effectiveness of managing the environment, it pointed to a number of areas that
have improved. They include in the National Representative System of Marine Protected Areas, so the declaration
of those areas, and in the management of the Murray-Darling Basin and some of the material changes which are
coming through or starting to come through—the first tentative signs of improvements in water quality and
biodiversity in the Murray-Darling Basin. That will take some time. There is also commercial fisheries and
Australia's sustainable fisheries management approach in Commonwealth fisheries, which is led, obviously, out of
the Department of Agriculture and Water Resources and the Australian Fisheries Management Authority but in
close partnership with our department; the management of shipping vessels; and offshore oil and gas operations.
So they are a few of the areas that have improved. Also, there are significant improvements in the understanding
and data around the environment.

Senator URQUHART: So they are what you say are improvements. Is there anything new in the report?

Mr Thompson: It does—I am trying to understand your question around what is new. I can touch on some of
the vulnerabilities that are identified in the SoE.

Senator URQUHART: I will go back to the point that Senator Moore was raising. Some of the comments in
the report are highly critical of the fact that there is lack of leadership, policy and follow-up policy. So I am trying
to establish what is: is there anything new? Are there new policies or are there new proposals from government?
Is that arising out of that report?

Mr Thompson: Sure. In the answer I gave to Senator Moore a bit earlier, I indicated that there was not a
general response by the government—a policy response—to State of the Environment 2016 but that we now
reflect on the findings of the report and the material there in terms of the policy advice that we give to
government, the spending advice that we give to government et cetera. So, in part, the government's decisions in
recent times around threatened species and providing additional resources for the Threatened Species Strategy—I
think it is a further $5 million being provided to that—reflects some of the findings in the SoE around the
continued pressure on threatened species across the nation and, in particular, the decline in the status of small
mammals in the Australian environment.

So that is one example. Of course, we have an ongoing and comprehensive response to the issues around reefs
and, in particular, the Great Barrier Reef. SoE points to some of the challenges that are manifest for coral and
fringing reefs and the issues there and the crown of thorns starfish in particular in the Great Barrier Reef. There is
no general response, but there is an ongoing response by government in terms of its policy settings, including—
Senator URQUHART: Are you saying that there will be new policies as a result of the State of the
Environment report? Is that what I am hearing? I do not want to put words into your mouth, but I am trying to
actually—

Mr Thompson: That is fine—thank you for clarifying. I am not saying that there will be policy. I do not
expect there will be policies that the government will announce and the government will say, 'This is because of
State of the Environment'. What I expect is that there will be advice to government and continuing to go to
government which reflects the findings of State of the Environment 2016 and which the government will consider
and make decisions on accordingly. In the budget, of course, there was a commitment to a further $1.1 billion for
the National Landcare Programme over seven years. That is obviously a major contribution to a number of the
land management and biodiversity challenges which the SoE identifies.

Senator URQUHART: That goes nicely into my next line of questioning about Landcare. It is $1.5 billion,
did you say, or $1.1 billion?

Mr Thompson: It is $1.1 billion over seven years.

Senator URQUHART: Is that a decline since 2013?

Mr Thompson: We might take that up in more detail in program 1.1, when the officers will be at the table for
that discussion.

Senator URQUHART: But can you as a department tell me if that is a lesser amount than in 2013—the $1.1
billion over seven years?

Mr Thompson: I am probably not going to satisfy you with this answer, but it depends a bit on how you
measure—over what time period you measure it. That seven years obviously includes some additional funding in
the current financial year—2016-17. It includes the commitment that was made in the Mid-year Economic and
Fiscal Outlook as part of the deal with the Australian Greens around Indigenous Protected Areas and the National
Landcare Programme. It is the commitment for the rolling cycle of National Landcare Programme funding.

Senator URQUHART: When the department is calculating environment funding, what components does it
include? What are the components that you include when you are calculating that funding or is that something for
outcome 1.1?

Mr Thompson: In the broad, when we are talking about environment funding, we would include all of the
domains that certainly the portfolio works on, including climate change, the Commonwealth Environmental
Water Holder for freshwater and inland waters and the work of the Director of National Parks as well as some of
the spending programs and, of course, the government's significant investment in the Antarctic program through
the Australian Antarctic Strategy and 20 Year Action Plan, which was announced last year. Again, I do not have a
number that reflects all of the—we do not produce an environment budget statement, as we used to do many years
ago, which tried to capture all of those spending lines. When we think about spending on the environment, we
would include all of those items and probably more—others that are undertaken in other portfolios, including the
agriculture department on biosecurity, for example.

Senator URQUHART: Just as an example, if you are attending an international conference or talking to
stakeholders, what figure does the department use for environment funding? Is this separate from energy and
climate change programs? How do you calculate it?

Senator Birmingham: I think that is a fairly broad question. For example, we spoke before and I am sure we
will come back to talking about the collaborative approach in areas like the Great Barrier Reef, where, if we are
talking to international stakeholders, we would be wanting to talk about the collective Australian effort—our
work together with the Queensland government and other relevant stakeholders. So it would depend very much
on the circumstances, I guess, and who we were talking to as to how you would say, 'This is Australia's focus on
this particular area of the environmental effort'. If we were talking about emissions reduction and emissions
mitigation, we would talk about the suite of policy measures that are relevant to that, which are in part the
Emissions Reduction Fund, in part the impact of the Renewable Energy Target, in part the work of Clean Energy
Finance Corporation and in part the work of ARENA. There is a range of different factors that come together
across these policy settings.

Senator URQUHART: Does the government know what its environment budget is?

Senator Birmingham: Do you want to know what we spend as a government—as the Commonwealth
collectively across environment? Sure.

Senator URQUHART: As a total. I guess what are the examples other than climate change?

Senator Birmingham: Including other stakeholders or just what the Commonwealth spends?

Senator URQUHART: What is the government's budget on the environment and what are examples other
than climate change? What are the sorts of things that you would budget into the budget?

Senator Birmingham: You heard some of them this morning—Landcare programs, water recovery and
restoration programs and the Great Barrier Reef. But, if you want an aggregate, I am sure that the department can
take that on notice and aggregate the different agencies into some sort of sum. To be honest, I am not sure that
will tell you terribly much, because there is a range of different priority areas across which that spending occurs,
some of which overlap with one another but also many of which then overlap with efforts of other jurisdictions
and other stakeholders too.

Senator URQUHART: Dr de Brouwer, can you answer that question about the environment budget?

Dr de Brouwer: I think that, frankly, as the minister said, if you are looking for the broad impact, both the
direct and the indirect impact, the very rough approximation is to sum up the spending across different activities
in the department and in portfolio agencies. But I can give some examples of that. The Emissions Reduction Fund
is for emissions reduction, so it has a climate change mitigation benefit. But almost all of those programs have
economic or social co-benefits. Three-quarters of the funding goes to farmers and that helps farm productivity.

The savannah burning programs are an important element of economic and social benefit to Indigenous
communities. So there is a range, even if it is—it is very hard to find a clear delineation between what is narrowly
environmental in terms of natural resources and what is the broader environment benefit. I do not have a number
around summing up the different activities, but we can do that. We will take that one on notice. That is a very
rough approximation.

Senator URQUHART: Minister, can you clarify when the review of the National Landcare Programme is
likely to be complete and when recommendations and conclusions might be available?

Mr Thompson: The Landcare Programme review is currently underway. We do hope to have that finalised in
the coming months.

Senator URQUHART: In coming weeks?

Mr Thompson: In the coming months.

Senator URQUHART: Does that mean the next couple of months?

Mr Thompson: I think that is a reasonable assumption.

Senator URQUHART: Okay. So probably before the—early in the new financial year?

Mr Thompson: I think that is a fair assumption.

Senator URQUHART: Again, this is to the minister, but if you want to flick it then I am sure you will. Can
you confirm that the allocation of money apportioned to the environment portfolio from the $1.1 billion in the
rollout of the National Landcare Programme moneys referred to in the budget will stay the same as previously?
Will it rise or will it fall between now and 2023?

Mr Thompson: I think we can deal with that issue in program 1.1, when the officers are here.

Senator URQUHART: So that is in program 1.1 as well?

Mr Thompson: Yes.

Senator Birmingham: Just for the record, Senator Urquhart, it is $1.1 billion, not $1.1 million.

Senator URQUHART: I did say billion. I have a few more questions on that, but I think probably program
1.1 would be the area.

CHAIR: The committee will now suspend for morning tea.

Proceedings suspended from 10:34 to 10:51

Great Barrier Reef Marine Park Authority

CHAIR: We will resume. I call officers from the Great Barrier Reef Marine Park Authority. Dr Reichelt,
welcome back. Would you like to make an opening statement?

Dr Reichelt: Thank you, Chair. I would like to make a brief update for the committee on the state of the Great
Barrier Reef and our management. First, I would like to acknowledge the traditional owners of the Great Barrier
Reef and their continuing connections to the land and sea country, and I pay respect to the traditional owners of
the land we are meeting on today. The Great Barrier Reef has been impacted on by two significant environmental
incidents this year: an unprecedented second consecutive year of coral bleaching on a mass scale and Tropical
Cyclone Debbie—the tenth severe category cyclone to cross the Reef since 2005. The Great Barrier Reef Marine
Park Authority is focused on understanding and responding to the impacts of these events.

To confirm the extent and severity of the 2017 bleaching event, this year authority staff took part in an aerial
survey conducted by James Cook University. The 2017 bleaching footprint was mainly in the central region,
differing from the 2016 event, which was mostly in the northern region. On 28 March, Tropical Cyclone Debbie
crossed the coast at Airlie Beach, causing significant damage in the Whitsundays region of the marine park. The
authority has been working closely with our partners and tourism stakeholders in the region to assist recovery in
that area. This includes providing permissions for active intervention. This is a fairly new thing for us. There is a
short window when, if you turn a coral over, it will survive. It is a small thing. We lifted the normal ban on
touching coral for the couple of weeks that that opportunity was there. A lot of the operators did that and were
grateful for the recognition that there was something that they could be doing. We also returned washed-up
boulders to create structure for settlement of new corals into the deep water. The massive waves were lifting
corals the size of small vehicles up and out of the water. They would be dead, but their structure is important, so
we rolled them back into the ocean.

We have a marine tourism contingency plan which was active and we were responding to the immediate issues
of the people in that region and the tourism operators, including by making contact with Maritime Safety to get
the safety radio channels working. They were wiped off the top of the hills in the region. Safety of life is a high
first priority in that response. Maritime Safety Queensland did do that very well.

Going back to the larger issue, the international reports now confirm that large-scale coral bleaching leading to
loss of coral, as in death of coral, is occurring globally throughout the tropics. This week I inspected corals at
Lizard Island off Cape York and Moore Reef near Cairns. Both of those reefs are experiencing strong effects from
coral bleaching, as are many others. Coral bleaching and any subsequent death of the corals are directly related to
heat stress—protracted elevation of sea temperatures above the long-term average levels. When the water cools
after summer, corals may recover. Some may recover and others are killed by that prolonged stress.

The global oceanic heatwaves are consistent with the steadily rising ocean temperatures predicted by climate
change modelling and global warming of the ocean. This signals a large-scale shift in the Reef's ecosystem.
Putting protective measures in place and letting the Reef recover in a natural way is no longer sufficient to
prevent the long-term decline that we predicted in 2014. We need to take more direct actions in the marine park to
help the ecosystem to retain its key natural attributes. To this end, the authority is hosting a Great Barrier Reef
summit meeting this week in Townsville, which will bring local, national and international experts in marine park
management, science and coral reef resilience together with partners and stakeholders from the region to discuss
the state of the Reef and evaluate both the existing methods we are using to protect the Reef and options for new
approaches. The authority is doing all it can to build the resilience of the Reef through implementing critical onground
actions to address impacts over which we have control. This includes strengthening measures to ensure
compliance with the marine park rules, preventing illegal fishing and supporting the control of coral-eating crown
of thorns starfish.

The protective measures in the marine park play a large role in delivery of the Australian and Queensland
governments' Reef 2050 plan. This is the 35-year management framework supporting the health and resilience of
the Reef.

We are managing an ecosystem undergoing rapid change as a result of global ocean warming. Strong action is
now required to ensure a healthy Reef for future generations. We are doing what we can to conserve and protect
the Reef, but the most serious risk to the Reef comes from outside of the marine park. This has been widely
reported, and I am sure that the committee members know this. The climate change impacts on coral reefs are
predicted to worsen and critically affect the survival of coral reefs globally without the strongest possible efforts
to reduce greenhouse gas emissions. Decisive action on reducing the levels of carbon dioxide and other
greenhouse gases in the atmosphere is essential to ensure the survival of the Great Barrier Reef and leave this
legacy of stewardship for future generations. Thank you.

Senator CHISHOLM: Thank you for the update on where the Reef is at. I am just interested in the summit
meeting that you talked about and new approaches. I understand that the summit has not happened, but is there
anything you would be able to elaborate on for us?

Dr Reichelt: I am very much looking forward to the meeting for the detail of it. The principle that we have
gone into this with is that we have a philosophy of conservation since the US invented the national park.
Essentially, it is managed for nature's resilience—in other words, get out of nature's way and prevent damaging
use. In a case where there is the big external threat of global change, we have come to the realisation that we need
to take action to boost the resilience and abundance of corals and take rehabilitation steps that previously we
would not have thought were needed. What we need is all of the actions, if feasible and do not do any harm—do
no harm—that we can take, such as encouraging more rapid regrowth of corals by—are there any other protective
measures, such as changing the anchoring locations; are there ways of encouraging heightened settlements of
coral spawn. We do some of these things now, and we want to see if we can do them better. But, for things like
whether you can enhance settlement rates of coral, we do not know that or whether it is a sensible thing on a large
scale.

I have said publicly that I want everything on the table. No idea should be seen as off-limits until we look at
and assess the risk and the likelihood of a return on the effort. So things like enhancing coral recovery are
definitely on our agenda for the coming week. We know that people around the world are trying it now in
different ways. We know that there is a massive natural selection process occurring now in corals worldwide.
What is it that makes some corals survive? Where are the thermally tolerant corals? We know that corals can
adapt. We do know that in millennia past they have had millennia to do that adaption. There are corals that are
adapted to higher temperatures than we are experiencing now, but they are there and they evolved in that place.
Can they move, for instance—can they migrate? Will we see migration of thermally tolerant species? We do not
know. We want to discuss that with the experts and just establish whether there is anything we can be doing that
will help the Reef while the processes of mitigation of greenhouse gases, which is a long process, are undertaken
by the global community.

We do know that they have had millennia to do that adaption. There are corals that are adapted to higher
temperatures than what we are experiencing now. But they are there and they evolved in that place. Can they
move? Can they migrate? Will we see migration of thermally tolerant species? We do not know. We want to
discuss that with the experts and establish if there is anything we can be doing that will help the reef while the
processes of mitigation of greenhouse gases—which is a long process—is undertaken by the global community.
Senator CHISHOLM: In your opening address you mentioned the independent expert panel report. Did you
quote from that or was it just a sort of commentary on it?

Dr Reichelt: I did not quote from it. I am a member of that panel, and it met quite recently. I do not know
when it was published but I have seen media commentary on it. I was not quoting directly from it, but I was a
member of the group that came to those conclusions.

Senator CHISHOLM: In that, Professor Chubb is quoted as saying 'long-term damage may be irreversible if
action is not taken now'. As a member of that panel, do you agree with Mr Chubb's analysis?

Dr Reichelt: In a general sense, that is true. It is more or less what we said in 2014 in our outlook report. In
the covering letter to the minister and the parliament in making that submission, I wrote that the condition was
poor and likely to worsen. When that was being written we thought there was more time for the climate impacts
to be dealt with than we have now, or than is apparent from the current intense bleaching. We are also worried
about declines in water quality. There has been a massive effort to improve those, which is ongoing, and will take
a long time as well. Why I agree with Professor Chubb's comments is that it is not just about climate change.

There are other pressures on the reef and things we can do something about more immediately.

Senator CHISHOLM: Linking the comment you made about the summit later this week and the report
saying 'if action is not taken now', do we need to change our approach?

Dr Reichelt: As the park managers we are declaring now that we want to change our approach. We want to
examine every possible action that could be taken that will build the Reef's resilience in the short to medium term,
while the things that we are not charged with, such as the global climate change mitigation, are occurring at full
speed. They will still take a long time. I am distinguishing our role as marine park managers and saying that we
think we need to change our approach. As for what that looks like, let us see what the community we are pulling
together later this week comes up with. I am looking forward to reporting back to you on that.

Senator CHISHOLM: In the report, Mr Chubb is reported as saying that we need to push for an international
response that could lead greenhouse gases to no more than 1.5 degrees. Do you agree with him on this?

Dr Reichelt: I will leave the commentary on the international response to other parts of the portfolio. In terms
of 1.5 degrees, yes, I do—for a couple of reasons. We have probably passed the safe levels for coral reefs already.
We saw signs of a lessening of the density of coral skeletons in a systematic way in the mid-1990s. That is
measurable in hundreds of coral cores from around the Great Barrier Reef and elsewhere. People talk in very
general terms about targets like the 1.5 degrees. Remember that those are results of models with errors around
them. Two degrees or 1.5 degrees does not mean that it is that every year. It means that it could be higher or lower
as of the local variability. The 1.5 degrees is the level at which climate physiologists and biologists, through their
experimentation, think that there would be good survivorship of thermally tolerant coral glades or genetic types if
the limitation to that level was achieved. We have seen a 0.7-degree rise in the last century. I draw the public's
and the committee's attention to the fact that the unprecedented back-to-back bleaching we have seen is now
occurring on the strength of a large fraction of one degree—0.7 degrees. It heightens the risk each year that the
temperature of the underlying ocean increases. That is a very long answer to say that, yes, I agree with 1.5 degrees
as the best goal, given the reality of the massive change that needs to occur in CO2 levels.

Senator CHISHOLM: Unfortunately, I have never been part of an expert panel, but I presume that they work
on a consensus basis in terms of how the report is put together.

Dr Reichelt: Yes. The communique is a short commentary of a much more detailed discussion which is due to
be reported in the long run. The secretariat for that group rests with the department. On the process of how they
work, you would be better asking them. But as a member of it, we have no constraints on how we talk and there is
general consensus on the urgency and the 1.5 degrees. I can confirm that that was a widely held view.

Senator CHISHOLM: Has the minister been informed and has the minister responded?

Dr Reichelt: You would have to ask the minister, in terms of the processes of the committee's working.

Dr de Brouwer: In general terms, the independent expert panel put out its communique, but that more
detailed report is going to the minister, and that is for his consideration along with that of the Queensland minister
at their ministerial forum in July.

Senator CHISHOLM: So the minister has not read the report yet?

Dr Reichelt: I am sure that the minister would know about the communique.

Senator CHISHOLM: Can I clarify, Dr Reichelt, one thing that you said. When you talk about a change of
approach, is that within the current Reef 2050 Plan? Are you talking about change within that current framework?

Dr Reichelt: Yes. The current framework anticipated these events. The way that framework is described and
its relationship to climate change no doubt will come under close scrutiny at its review next year. It was designed
to be adaptable and be reviewed. It is due for review in 2018. The long-run reporting and cycle of change for that
plan will be every five years and is designed to come out after our five-yearly outlook report. It was anticipated
that it would need regular updates, and that is planned for next year. But the actions that I am talking about are not
excluded by that at all.

CHAIR: So it is all within that.

Dr Reichelt: The framework still works.

Senator Birmingham: My understanding is that the communique, which Senator Chisholm has been referring
his questions from, does refer to the Reef 2050 Plan as being a strong foundation. Of course, that is the approach
we have taken in building something with the Queensland government that ensures there is a cross-government
framework and plan upon which integrated action happens through GBRMPA and through other agencies of both
governments and other stakeholders to deal with these challenges.

CHAIR: Going back to Senator Chisholm's question, do you have an answer for that? Has the minister seen
the communique?

Mr Oxley: The ministerial forum that was held at the end of last year asked the independent expert panel to
provide it with some advice on what further action could be taken to deal with the impacts of coral bleaching.
That obviously has been given a much greater emphasis as a consequence of the bleaching event continuing into
2017. The independent expert panel met a couple of weeks ago, as Dr Reichelt has outlined, and discussed the
nature of the advice that it would be providing to the two ministers—Minister Frydenberg; and Minister Miles,
the Queensland Minister for National Parks and the Great Barrier Reef. The panel issued the communique, which
has been discussed already this morning, and it is now in the process of formulating its comprehensive advice to
ministers. That comprehensive advice will be provided to ministers for their consideration at the next meeting of
the Great Barrier Reef Ministerial Forum in mid-July.

Senator Birmingham: Just to be clear, Senator Chisholm, Minister Frydenberg has received, seen and
reviewed the communique, which provides, obviously, the substance of the questions we are having here. The
more detailed body of work is actually one still to be completed and signed off by the members of the expert
panel. It has certainly not been provided to Minister Frydenberg. He has been unable to read it because it is yet to
be a completed document signed off by the expert panel.

Senator CHISHOLM: In terms of a response to what has been put out there already, has there been any
government response to that?

Senator Birmingham: You have heard the types of mechanisms that are already in place for the ministerial
discussions and work with Queensland and other stakeholders to be able to respond to that. That will be further
undertaken when the detailed findings are provided in the not-too-distant future.

Senator CHISHOLM: Given the nature of the report, is that going to lead to a discussion about a change in
Australia's approach to the Great Barrier Reef?

Senator Birmingham: Our approach has been to build what is acknowledged to be a strong framework in the
Reef 2050 Plan. We will continue to build upon that, and that includes around 151 different actions within a
strategy that is worked upon collaboratively with the Queensland government. It includes around $2 billion worth
of ongoing investment and action. In addition, there is the $1 billion within the CEFC Reef Funding Program for
particular investment in climate change and water quality projects that can have impact in relation to the Reef
region, key catchments, the reduction in nitrogen run-off, as well as assisting with climate change mitigation
activities. We will continually work to improve these policies and processes, advised and informed by such work.
As you heard, there is a formal review mechanism in place for the Reef Plan. Obviously, these findings will
inform that review and, of course, ministers will have their discussions together about further actions that need to
be undertaken.

Senator CHISHOLM: Does the quote from Professor Chubb which says 'long-term damage which may be
irreversible unless action is taken now' concern you?

Senator Birmingham: Obviously we are sufficiently concerned for the Reef to have undertaken all of the
types of investment and responses that we have to date, and that we continue to undertake. In terms of the impact
of climate change specifically, ultimately the Paris Agreement is the most appropriate forum for approaching
climate change mitigation and emissions reduction. It can only be accomplished with a concerted global effort. It
is not something that Australia can do in isolation. So we work through that Paris Agreement. We have committed
as a nation to emission reduction targets that seek to achieve and halt climate change well below two degrees,
consistent with the types of ambitions that Dr Reichelt acknowledged earlier. That is the forum for overall efforts
on climate change that we focus our efforts on. Then, of course, there are a range of domestic policies that we
pursue to ensure that Australia meets and, as we have historically done, exceeds its commitments in relation to
those international agreements and commitments.

Senator CHISHOLM: I think, Dr Reichelt, you identified that climate change is a significant threat, but there
are other threats as well. Could you update us on the respective role of curbing land-based pollution and sediment
flows and the crown-of-thorns infestation—how progress has been made on those issues?

Dr Reichelt: If I could split those two, because we have a strong role in assisting the considerable government
investment through the Reef Trust to support crown-of-thorns control. I will speak to that briefly and then ask my
colleagues who are handling the landward side to speak to that in a moment. The crown-of-thorns control now has
two vessels operating. I have to say that they are making a huge difference. They have expanded from 22 to 50
reefs that they are operating on. With two vessels now, they really are proving that an at-scale control of the
unnatural outbreaks of this predator seems highly feasible.

So that people understand, the crown-of-thorns reaches densities of thousands of starfish in small areas. They
will denude a reef of 50 per cent coral cover to two per cent coral cover in 18 months to two years. For every one
of those unnatural outbreaks that can be controlled, you are saving those corals from having to go through a 20-
year recovery cycle. The question we are asking now, and we will be asking this at the summit coming up later in
the week, is: what are the multiplier effects? By targeting the source aggregations, can we be cleverer about
integrated pest management, taking that into the ocean? We do not want thermally tolerant corals that survive the
bleaching to then be eaten by a starfish that is being fuelled by nitrogen run-off. The two things that you asked
about are very much linked. I want to reassure the committee that I really strongly support the work being done
on crown-of-thorns.

Having worked on it for 40 years, I never would have thought it was controllable until the last few years. And
it is an innovative program. The government has funded some research to look at better ways of doing it, and it
keeps getting better. I am very happy with the work that is being done there. The crown-of-thorns at the moment
is moving in a wave, as it has done since the 1950s, with the east Australian current. It is now concentrated
between Cairns and Mission Beach, that sort of area. It will move into the Townsville area. It will move to Bowen
unless it does something different to what it has done for the last 50 years. Then it sort of stops about there. We
do not know why. We think it is either temperature driven or it is much farther away from the nutrients. That is a
good program. It is strongly supported. We have the details of the funding and the staff.

I also commend the marine park tourism operators who are delivering it to the Reef and Rainforest Research
Centre. They have a strong Indigenous traineeship working within that program. These young people come off
there with commercial diving qualifications and job prospects. I know that is not the purpose of it, but it has a lot
to recommend it. We will be looking hard at whether we can make it better.

I will ask my colleagues to talk about the run-off.

Ms Parry: In addition to what Dr Reichelt has outlined in terms of the crown-of-thorns, for which the
Australian government has contributed $18.8 million since 2014-15 to support crown-of-thorns control and
research, we also put in significant investments in the catchment side through the Reef Trust and, previously,
through the Reef program. Most recently, phases 4 and 5 of the Reef Trust were rolled out to address, again,
highly targeted industries and regions. This particular round focused very much on gully and stream bank erosion
across the highest threats and targeting the greatest sources of sediment discharge into the Reef regions.

In addition, we have also leveraged considerable financing through phase 5, with partnerships with Greening
Australia. In total, the Australian government contribution of $7 million was matched by Greening Australia to
bring a total of $14 million into wetland rehabilitation. On the back of that, the Reef Trust also was involved in a
partnership deal with Msf Sugar, looking at lowering and improving behavioural practices and farming practices
for the Msf Sugar farms that feed into that particular mill. The Reef Trust invested $4.5 million and Msf Sugar is
investing $12.8 million. Those are the latest Reef Trust 4 and 5 investments.

Senator CHISHOLM: Dr Reichelt, are these issues covered off in the Reef 2050 Plan?

Dr Reichelt: Yes, in great detail.

Senator CHISHOLM: Similarly, in regards to climate change as well?

Dr Reichelt: The climate change risk is the global one I talked about, and mitigation. That is not part of the
Reef 2050 Plan. It is acknowledged in it as a key issue and a key risk, but it is a separate activity.

Senator CHISHOLM: So there are no actual policies in the Reef 2050 Plan about curbing climate change?

Mr Oxley: If I may, you are going to a question about the Reef 2050 Plan, and the department is responsible
for the Reef 2050 Plan. So, if you do not mind, I will give you the initial answer. The Reef 2050 Plan
acknowledges right up-front that climate change is the single biggest pressure and threat facing the Great Barrier
Reef. But the government was also very clear, in partnership with the Queensland government in developing the
Reef 2050 Plan, that its focus was on improving the health and resilience of the Reef so that it was best placed to
deal with the pressures associated with climate change. But the place for climate change policy to be prosecuted
was not through the Reef 2050 Plan but through the government's engagement in the Paris Agreement in the
UNFCCC process. That is very clear from all of the documentation associated with the Reef 2050 Plan. So while
climate change is acknowledged as the single biggest pressure facing the reef, the plan also recognises that the
Reef 2050 Plan is not the place to prosecute climate change policy outcomes.

Senator CHISHOLM: But you would have thought, given its impact on the reef, that it would have
recommendations about how this is best perused.

Senator Birmingham: I think the Reef 2050 Plan makes it clear that action to address climate change needs
to be a priority. It is a priority and we implement that priority through the Paris Agreement and then through the
different policies that we deliver to meet Australia's commitment under the Paris Agreement. But, though the
issues of climate change are very important to the future of the Reef, the Reef is not the only area in which those
issues are important. Climate change takes on a priority in other ways as well as in relation to the Reef. Of course,
as we emphasised before, it requires a global effort to actually sustain serious action in relation to climate change.
That is where the focus lies. It is identified up-front as a priority and then government takes actions to address that
priority in a range of ways. But the Reef 2050 Plan goes specifically to a range of actions and strategies that are
applied and coordinated across the Australian and Queensland governments, in different agencies and
stakeholders, to deal with a range of other factors that impact on the Reef—but also areas of mitigation strategies
that directly relate back to what some of the impacts from climate change may be.

Senator CHISHOLM: But it does not recommend policies on the most significant issue facing the Reef.

Senator Birmingham: No, I do not accept that, Senator Chisholm. It does recommend that there needs to be
actions to address climate change and it does recommend actions in relation to adaptation and issues that relate
specifically to factors around climate change as well.

Senator CHISHOLM: It does provide recommendations in those other areas, it just does not do so in this
climate change area.

Senator Birmingham: Officials may want to talk specifically to the content of the Reef 2050 Plan. They can
probably cite some of the words within it. It identifies that action is important in relation to dealing with the issue
of climate change. It also, in terms of the different plans and strategies and so on that are outlined in the Reef
2050 Plan, includes plans and strategies that relate to the specific areas of adaptation and action that can be taken
in the context of the Reef as it relates to the climate change threat and other threats that the Reef faces.

Dr Reichelt: The focus of the Reef 2050 Plan is all about the resilience of the system in the face of climate
change. That is how it is couched, right up-front. Some of the interlinkages between water quality, crown-ofthorns,
coastal development and even over fishing areas—we interpret all of them in light of the resilience of the
system in the face of climate change. We strongly support some of the work mentioned earlier on water quality,
because it is known that corals living in water with elevated sediment nutrient are more prone to bleaching, for
instance. So the act of cleaning up coastal waters is a way of building the resilience of the system. There is four
times the amount of sediment and nutrients coming into the coastal waters than there was pre 1700. I strongly
support the Reef 2050 Plan because it puts the focus on building resilience. That is not meant to detract from or
deny the importance of the national and international policies on climate change, which I have always
acknowledged are extremely important. They are addressing the primary external risk to the Reef. It is just that
the actions that we can take in the marine park are to build the resilience of that system in the face of that. That is
how I see it.

Senator CHISHOLM: Dr de Brouwer, I am interested in the vegetation land clearing that we have been
seeing in Queensland. I had the opportunity to ask questions at our last estimates meeting. At that, you mentioned
that under federal law the EPBC Act applies. Has the federal law been used around any of the land clearing that
we have seen in Queensland in recent months?

Dr de Brouwer: The EPBC Act will apply in relation to matters of national environmental significance,
which will come down to the effect on threatened species and on World Heritage, particularly the Reef. And then
it will be a matter of whether that particular economic activity or land clearing will have a significant impact on
threatened species or the Reef. It comes back to the regulation. It comes back through that mechanism. We can
back in outcome 1.5. That is one of the places to talk about the regulation side of it.

Senator CHISHOLM: You generously let me ask these questions last time. That was your mistake!

Mr Knudson: This has been a point of questioning for a number of different estimates hearings. You may
recall that we were looking at 54 property owners in Far North Queensland. Forty-six of those 64 have been
notified that they are not going to have a significant impact on matters of national environmental significance and
therefore will require no further assessment. Six property owners have been advised that their approval is likely
required for the land clearing that has been proposed. There are two other properties that we are currently
continuing to look at, but we can talk about that in more detail, as the secretary said, in 1.5.

Senator CHISHOLM: In regards to World Heritage issues, is the World Heritage Committee's technical
advisers report on the Great Barrier Reef available?

Mr Oxley: No.

Senator CHISHOLM: Do we have any intelligence as to what is in the report?

Mr Oxley: Not a great deal of insight. The answer to that question will be known on 2 June, I suspect. That is
the date for the next and final release of documents for the World Heritage Committee meeting in Krakow,
Poland, at the beginning of July.

Senator CHISHOLM: In terms of the context of the bleaching events that Dr Reichelt has outlined, is there
any idea of what impact that is going to have and whether the World Heritage Committee might consider placing
the Reef on the endangered list?

Mr Oxley: I think we are entering the realm of speculation about a possible outcome. Obviously the World
Heritage Committee's technical advisers are very attuned to what is happening on the Great Barrier Reef in terms
of coral bleaching, as they are attuned to the coral-bleaching impact on other World Heritage listed coral reefs.
So, our understanding from communication with IUCN and the World Heritage Centre is, I think, that, of just
under 20 World Heritage sites where coral reefs are one of the aspects of outstanding universal value, 13 or 14
have experienced coral bleaching of varying degrees of severity in the course of the continuing coral-bleaching
event that has been ongoing for two or three years now. That is really bringing into focus for the World Heritage
Committee and the advisory bodies the question about how the World Heritage system should address coral
bleaching writ large, rather than through the lens of one property.

Senator CHISHOLM: Has any decision been made on who will represent Australia at the World Heritage
Committee meeting in July?

Mr Oxley: Currently the delegation composition is me as co-head of delegation, our ambassador to UNESCO
and definitely one of the ambassador's administrative staff, who has been working in the World Heritage system
for about 20 years and knows it inside and out. I will have two of my staff with me to prosecute our engagement
in the wider business of the World Heritage Committee. There is a substantial body of work around financing and
administration of the system and its operations which we engage in deeply every year. I also expect that I will
have Ms Parry from Reef Branch, and Dr Reichelt. That is the proposed composition of our delegation at this
point in time.

CHAIR: Senator Chisholm, you have had just over 30 minutes, so I will go to the Greens senator and then we
will come back.

Senator WATERS: I will continue on the line of questioning about the Reef 2050 Plan. The independent
expert panel made a clear recommendation that the reef plan be amended to include climate change adaptation
and mitigation actions. Minister, perhaps this one is for you. Is that under genuine consideration by this
government?

Mr Oxley: If I might, rather than the minister, and Ms Parry may wish to expand on my answer. There is a
review of the Reef 2050 Plan scheduled for 2018. We are working on the expectation—and it is the express
expectation that has similarly been conveyed by the independent expert panel—that the sort of issues that they
have raised would be considered in that review of the Reef 2050 Plan.

Senator WATERS: Again, can I ask the minister, is the government actually considering putting climate
change mitigation in the reef plan?

Senator Birmingham: Senator Waters, as we have made pretty clear in the discussions here already, the
importance of climate change mitigation as a factor is made clear in the reef plan. The government then takes a
range of actions and strategies, driven at an overall level via the Paris Agreement, to give effect to actions around
climate change mitigation.

Senator WATERS: So you won't be putting those in the reef plan?

Senator Birmingham: If you would like us to put in the reef plan details of the Paris Agreement, details of
our investment through the Emissions Reduction Fund, details of our investment through the Clean Energy
Finance Corporation, details of the workings of the renewable energy target and its impact and details of our
investments through ARENA, then they are things we could go away and think about. But the Reef 2050 Plan, as
you have heard from officials, makes clear, firstly, the significance of climate change and the importance for
mitigation strategies. The government takes that seriously and is acting on it. But the objective of the Reef 2050
Plan is then, as Dr Reichelt has explained, around how we build the resilience of the reef, specifically and
particularly. Obviously the two work hand in hand. The more success we can have with mitigation strategies, the
more success we can have globally in that regard. That obviously helps. But at the same time, we seek to build the
resilience of the reef to a range of threats, of which climate change is a prime one.

Senator WATERS: You say you take that threat seriously, but you mention the ERF, which has not been
funded in this budget. It is essentially kaput, even though it used to be the centrepiece of your government's
purported climate change policy. You have not mentioned that your government is backing the largest coal mine
in the Southern Hemisphere and potentially giving it a billion dollars of taxpayer money. So how can anyone take
seriously your claims that the government is taking climate change seriously?

Senator Birmingham: As you are well aware, in terms of meeting Australia's future commitments in relation
to emissions reductions, policy work on that will follow the expert work that is being undertaken there to see how
we build on the success of the ERF in the future. Of course, that has put Australia in a position where once again
we will meet and exceed current targets. I am confident we will take the right policy responses to meet and
exceed the 2030 targets that Australia has set down as well. Those targets will be met and exceeded, regardless of
other development activities that may occur in other parts of Australia.

Senator WATERS: Those Paris targets put us on a trajectory of at least three or even four degrees of
warming. Some would say more. How much of the reef will have to die before this government addresses climate
change seriously and adopts some targets that might have a chance of saving what is left of the reef?

Senator Birmingham: The commitment that Australia has made under the Paris Agreement is to work
towards reductions that are well below two degrees. That is a commitment that we take seriously. Of course, we
have made commitments that, relative to Australia's economy and population, are amongst the biggest
commitments for reductions globally.

Senator WATERS: That is only because we are the biggest emitter. Anyway, moving on. Dr Reichelt, there
has been some research published by the University of Melbourne climate scientist, Dr Andrew King. The
temperatures we have just experienced this year resulted in the second successive event that was, as you said,
unprecedented in the reef's ancient history. He found that temperatures at that level would be 97 per cent likely
every single year by 2050 at the current rate of climate emissions. In your view, would the reef survive that level
of bleaching?

Dr Reichelt: If that were to occur, no-one knows precisely how much would survive, but it would be a
relatively small fraction of its current size. I think that that is the consensus amongst the reef scientists that I
know. The aim, of course, is to not let it go on the current trajectory.

Senator WATERS: What level of temperature increase is consistent with the survival of the reef?

Dr Reichelt: I don't think we know that, really. I have not seen anything precise. I have seen some
projections—and these were made 20 years ago—of when back-to-back bleaching may occur, which was in the
mid-2030s—and it has occurred now in 2017. So I think that the general conclusion about reaching these levels
will have a dramatic effect on coral reefs. To be precise about where in the intervening range of temperatures
things would happen, I don't think we have that predictive capability at the moment. That shouldn't stop us from
taking it very seriously and looking to remain under two degrees—1.5 degrees is what my colleagues tell me is
the level at which a fair proportion of the diversity of coral would survive.

Senator WATERS: In the submission to a Senate inquiry that I instigated with the support of the Senate in
2014, GBRMPA called for a limit of 1.2 degrees. Do you still stand by the 1.2 degrees or have you gone up to 1.5
degrees now?

Dr Reichelt: I would have to recall the context of the question. In the context of 1.2 degrees, the least rise the
better from the current. We have already seen 0.7 degrees and we have seen back-to-back bleaching. 1.5 degrees
is the product of the Paris Agreement. Being a pragmatist and knowing that this is a global problem, the more
global consensus and common purpose that the international community can have for reaching it, that is what we
should support.

Senator WATERS: Has GBRMPA been asked to contribute, and have you contributed, to the 2017 climate
policy review that the government has long spruiked?

Dr Reichelt: We are sending in a submission on that.

Senator WATERS: Has that happened yet?

Dr Reichelt: Yes, it has.

Senator WATERS: Is that publicly available? Are you able to provide us with a copy of that?
Dr de Brouwer: I think we are just all receiving them, and it depends on whether people have asked for their
submissions to be confidential. But we can come back maybe in the climate change discussion around the exact
number.

Senator WATERS: Did you ask for it to be confidential?

Dr Reichelt: No, we send it to the review team. I am just not sure of what conditions they have around it. We
could check that.

Senator WATERS: If you could provide that, it would be wonderful. Moving on to water quality, Jon Brodie,
who of course is a preeminent water quality expert, has said:

It's been my life managing water quality, we've failed… The federal government is doing nothing really, and the current
programs, the water quality management is having very limited success.

The reef scientists on the Queensland government expert panel had called for a significant investment in water
quality, to the tune of about $8 billion. Clearly our water quality investments from this government and the
Queensland government is nowhere near that amount. Do you share the view that we need that serious level of
funding in order to genuinely address water quality issues in the reef?

Dr Reichelt: I participated in the panel discussions run by Professor Geoff Garrett. I don't think that I or
GBRMPA have the expertise to be very precise. But the thing about water quality for me is that it is the outcomes
that should be driving us rather than the actual dollars. I'm not thinking more or fewer dollars. More resources
potentially could help. I am seeing in a couple of projects, which are probably less than a few hundred thousand
dollars, quite substantial progress in consensus among farming now in the Tully area. That is based more on
behaviour change and is less dollar related. GBRMPA and I come at it from, what we need to achieve is the water
quality guidelines that we have established for healthy corals and are now being incorporated into water quality
improvement plans and Queensland legislation. I would have to defer to my colleagues in the department as to
how they are tracking towards it. But we have been very clear on what the reef needs.

Senator WATERS: Okay. Just on that, I hear you that you say outcomes are important but the outcomes
aren't being met. The target for dissolved nitrogen is a 50 per cent reduction by next year and an 80 per cent
reduction by 2025. At the moment we have only had an 18 per cent reduction. So we are clearly nowhere near
meeting that most important target. Is it your view that the water quality targets under the reef plan will be met?

Dr Reichelt: What I am aware of is that the ecologically relevant targets, which were sparked by Professor
Brodie some years ago, are being calculated, if they are not already calculated. Before I hand to my colleagues on
the detail of that, there is a need to accelerate the improved water quality. I don't want to give you the wrong
impression that I am happy with not meeting those targets. What I am saying is that we do need to do what we can
to achieve the improvement in water quality as soon as possible. But it is something where we are setting the
measures, and it really would fall to the department in terms of their catchment management and investments to
decide where and how much is needed. I can talk to the generality and certainly endorse the need, in any way
possible, to accelerate the reduction of sediment, nutrient and pesticides in the Great Barrier Reef Marine Park.
We would be 100 per cent supportive.

Senator WATERS: I will take that up with the department at a later point. Can I come now to the little trip
that the foreign minister took some diplomats on to the reef recently? I think there were 75-odd diplomats,
presumably at public expense. Was the GBRMPA involved in that visit?

Dr Reichelt: Yes, we were.

Senator WATERS: What was the scope of your involvement?

Dr Reichelt: Firstly, I was there with supporting staff. I gave talks to the diplomatic core—and these were
diplomats in Australia representing other countries and Minister Bishop—about the state of the reef. I gave a talk
on the vessel. I think we had four or five staff from the marine park authority assisting in education work in the
water for snorkelers. All but two or three would have got in the water and inspected Moore Reef. I found a high
level of engagement and interest in the diplomats. Of course, the Department of Foreign Affairs and Trade and
Minister Bishop were leading or designing the trip. But we played a very strong role in education and
interpretation.

Senator WATERS: On that need for education, the minister was quoted as saying:
Marine biologists from James Cook University and the Great Barrier Reef Marine Park Authority were optimistic the
bleached coral can recover and rejuvenate without any mass die-off.

Is that an accurate reflection of the information that you gave to the diplomats?

Dr Reichelt: We gave a full explanation to them and saw the corals. Some were recovering; some were not. I
can't comment. I was not there when the minister said it. I do not want to comment on a quote from a minister. I
think there was a very good recognition by the minister, her officials and by the diplomatic core of the seriousness
of the bleaching of the site and the fact that some corals do recover—and the more from my point of view the
better—and some don't. So there was no holding back in terms of explaining the real serious circumstances of it.

The reef we visited had a good representation of all the different types of coral bleaching.

Senator WATERS: How does the minister's belief that it can just recover without any mass die off square
with the reality that we have just had two mass bleaching events that have led to the worse ever coral mortality in
the reef's history?

Dr Reichelt: Again, I am not sure of the context of the minister's comments. There are corals that do recover.
There are many that don't. I really don't want to speculate on the background of the comment by the minister.

Senator WATERS: Could I suggest that you continue that education and perhaps correspond further with the
minister to disabuse her of that notion? It sounds like she has gotten a bit mixed up.

CHAIR: Senator Waters, I think it would be helpful if you could confine yourself to questions rather than
providing advice to officials.

Senator WATERS: Sure.

Senator Birmingham: I think as Dr Reichelt has acknowledged, coral can recover. You have given a quote
from Minister Bishop. None of us have the context of that quote. You have made your point, if you wish, Senator
Waters. Recovery is a reality. Obviously everything depends upon the circumstances and future climatic events
and so forth.

Senator WATERS: Yes. Perhaps Dr Reichelt can elaborate on that. Because recovery is not inevitable. It
depends on temperature dropping. And that is not what has happened. So, Dr Reichelt, if you could, as succinctly
as possible—

Senator Birmingham: I don't think I used the word 'inevitable', Senator Waters.

Senator WATERS: go to the recovery and the—

Dr Reichelt: I am happy to talk about the reef.

Senator WATERS: number of contingencies that are required for it to occur.

CHAIR: Senator Waters, I think Dr Reichelt is trying to answer your first question. Rather than having both
of you speak at once, it would be helpful to let him finish.

Senator WATERS: There is only one question there.

Dr Reichelt: If you look at the whole reef, 93 per cent of the reef, or 3,000 reefs, showed some bleaching.

Serious effects were felt last year in the northern third and this year in the southern third. In the southern half we
have seen about a 40 per cent increase in coral growth because it hasn't suffered bleaching and cyclones. The
resilience of this system is strong. It is difficult to get this across without sounding like I don't think the problem
is serious. I certainly do. But we are seeing large-scale oscillations in coral abundance over large areas. We hope
that the southern regions of the reef continue their growth and that there is a lack of these major impacts. It is 
likely they will experience them some time in the future. The long term trajectory for the reef is declining, but
with very large bouts of recovery and then impact. The past has been storms and crown-of-thorns. The future
appears set to be those two things plus coral bleaching. That is why we have to be conscious of all of the things
going on and not sort of simplify it to what proportion of corals recover or die, because it is occurring over 2,000
kilometres. I know you were talking about a specific trip to a specific reef, but the bigger picture is that there is
resilience there now. It depends on the frequency of these major impacts. The concern is that the frequency could
well be increasing and the recovery time will be insufficient. According to Andrew King, if the recovery time is
short, there won't be a lot of coral.

Senator WATERS: Exactly.

Dr Reichelt: That explains the bigger picture.

Senator WATERS: Thank you, that was nice and clear. Can you confirm for me the total coral mortality from
bleaching events this year and last year? My understanding is that it is roughly half now.

Dr Reichelt: I have seen those in the independent expert panel. The JCU surveys and the models they have of
this year were in the vicinity of 20 per cent, I think. So that is 30 per cent last year and 20 per cent this year. That
is a modelled outcome, which our preliminary observations say could well be true. We are relying now on the
long-term monitoring program that covers the entire reef, rather than just areas of impact. We will see the 2016
figures released within the next few weeks, I understand. 2017 will take until this time next year. Without wishing
to frustrate the public, it is a very large system and the best data comes from divers in the water at the moment,
rather than aerial surveys. I think the responsibility on us when we do say what we think is happening, we want it
to be credible and able to be relied on. What we are saying is that we agree with the 29 to 30 per cent last year.
That is due for release in a much more detailed way.

What we aren't doing now is trying to guess what the figures will be for this year. We know it is central and we
know that it is serious. But I do not want to be drawn on a specific figure for the current year's impacts.
Senator WATERS: But you said 50 per cent coral death in the total reef was roughly consistent. Thirty per
cent last year and 20 per cent this year.

Dr Reichelt: Yes. I am just being reminded that in the northern region there was an increase for three years of
a similar amount, so before the bleaching. It is a reminder not to think of these figures as the net amount of coral
on the Great Barrier Reef. Because there are quite big movements upwards as well as downwards. It will take
some years to see these long term trends, as it did in the eighties and seventies with crown of thorns.

Senator WATERS: With that level of specificity being achieved by divers rather than modelling, is that
something you have had a funding increase to do?

Dr Reichelt: Yes. We received the integrated monitoring reporting funding several years ago. We are
spending about 40 per cent of that on actual projects in the field.

Senator WATERS: What year did you receive the extra funding?

Dr Reichelt: It was 2015-16.

Senator WATERS: Since the two successive back-to-back bleachings, have you received increased funding
to do monitoring?

Dr Reichelt: We received $2 million a year. But it is not just for the bleaching response. It is for designing the
system that we will need to have in place for the long term—for future outlook reports, for the assessing of the
Reef 2050 Plan.

Senator WATERS: So it is not for actual monitoring, as such.

Dr Reichelt: No. We have—

Senator WATERS: Is it a case that if you don't look you wont find it?

Dr Reichelt: No, it's not. We have 100 park rangers in our cooperative program with Queensland. We have a
$6 million vessel that has been a radical multiplier for their capability to look at this system. The other thing is
that we have managed to harness a cooperative arrangement with James Cook and with AIMS, who also have
vessels. So there is a high level of cooperation and data sharing at the moment. That probably wasn't there 10
years ago. We are better equipped now than even after Cyclone Yasi in 2009. What we need to do is make sure
we are spending our time on resilience building—the things I spoke about earlier. Monitoring is something that, if
we do it now, we'd be getting ahead of it. These events take a year at least to know what has survived and what
hasn't. Corals that appear recovered can revert. We have to think of it not as a bush fire that we can go out to the
day after and measure what burnt and then count the damage. It is a longer-term process.

Senator WATERS: Sounds like something that needs a bit more dedicated longer-term funding.

Mr Knudson: Back in the 2016-17 Mid-Year Economic and Fiscal Outlook, the government provided an
additional $124 million to GBRMPA over 10 years.

Senator WATERS: Wow! That is so much over 10 years. Compared to the $27 billion to coalmining
companies in cheap fuel, it pales, I'm afraid. But thank you for that clarification.

CHAIR: Sorry, Senator Waters, I didn't detect a question in there. It was more of a long political statement. If
you have a question, I think that would be helpful.

Senator WATERS: I have a follow-up question.

CHAIR: Dr Reichelt, is there something you want to further clarify?

Dr Reichelt: Yes. Having now had our budget stabilised, it was a massive injection of resources that
otherwise would not have been there. So we have stable funding for our current staffing levels. The thing that I
didn't mention is that the Institute of Marine Sciences, from another portfolio, is funded and injects considerable
resources to this long-term monitoring program that has now been running since 1985. That is our core, reliable
source of information. We cooperate closely with the Institute of Marine Sciences. There are significant resources
available. We are able to move staff from one activity to another. In fact, we surge into the underwater work when
we need to. What we do not want to do is duplicate another agency like AIMS. Their data is very reliable.
Senator WATERS: So after those big staff cuts of a few years ago, you haven't been able to go back up to
higher levels?

Dr Reichelt: We are stable and are at a level that enables us to conduct these kinds of activities.
Senator WATERS: One final question. You mentioned the summit that is going to be happening this week.
Who is going to be attending that?

Dr Reichelt: I can't give you the list at this stage. I probably could with some notice. It is principally marine
park managers, practitioners, engineers and scientists with an expertise within the field of operations. There will
be some international experts with expertise in coral resettlement and accelerated growth of coral. There will be
some geneticists and colleagues from NGO's who are partnering on restoration of things like turtle recovery, for
instance. We are drawing the line at the marine park. We could run a parallel process, but that would be the
department's job to do in the catchments and above the high water. It is practitioners and management options.
Senator WATERS: Is there anyone from the tourist sector?

Dr Reichelt: Yes, there will be tourism industry representatives.

Senator WATERS: Anyone from the resources sector?

Dr Reichelt: I am not sure about resources. We keep in close touch with them, but I don't believe so.

Senator WATERS: Okay. If you could provide a list on notice of who is coming along, that would be
appreciated.

CHAIR: If possible, given the imminence of the conference, if you are able to get that list back to the
committee today, that would be helpful.

Dr Reichelt: Agreed.

Senator WATERS: Thank you.

CHAIR: Senator Whish-Wilson, you have five minutes and I will come back to you later.

Senator WHISH-WILSON: Yes, please. I have a couple of questions and then if you could come back to me
later on a different line of questioning. Dr Reichelt, if I can boil everything that we have heard today into just a
simple assumption that the reef recovery plan alone won't necessarily save the Great Barrier Reef over the future
if we see increased global warming. You said the strongest possible action was required to tackle emissions.
Given your work you do with stakeholders, especially the tourism industry, do they understand that the strongest
possible action is going to be required outside of the reef plan to secure the long-term health of the Great Barrier
Reef?

Dr Reichelt: They are calling for more action from us, especially in reef repair. They do understand. I would
say they would be our strongest supporters, especially those who have just experienced Cyclone Debbie. But it is
all of the tourism operators that I speak to . It is not outside the plan, by the way. These are actions and options
that we want to put into the framework of the plan. Just to clarify that.
Senator WHISH-WILSON: I don't think we have a direct response whether the mitigation side would be
included in the plan.

Dr Reichelt: No.

Senator WHISH-WILSON: So the answer is 'no'.

Dr Reichelt: But that is not what we are doing this week.

Senator WHISH-WILSON: The reason I ask that questions was that I understand that you are looking at
these short term solutions like pumping cold water into areas. I am talking about what is outside the scope of the
recovery plan. We went to great pains in the first ten minutes to talk about the Paris Agreement and global action.
Does the tourism industry understand that, as you outlined in your own words, the strongest possible actions on
that are what will save the reef in the longer term?

Dr Reichelt: My experience has been that they do. Our first climate change action plan had a tourism
component led by the tourism industry. I have asked our high standard operators if we could perhaps increase the
definition of high standard operation to include mitigation and other things that are more directly related to
climate change. My experience is that they very much do. They certainly understand the science and the source of
the risk to the reef through greenhouse gasses.

Senator WHISH-WILSON: I would just be concerned if some of the other things we are looking at doing to
help adapt to climate change might be seem by some in the industry as being long-term solutions to this problem.
My next question relates also to this statement about the strongest possible action. Do you see preventing new
coal mines from going ahead, like the Carmichael coal mine, and the export of that coal overseas, as the strongest
possible action on climate change?

Mr Knudson: Just before Dr Reichelt talks, I just want to clarify a point that I think Senator Waters made
which is just factually incorrect. It is that the existing Reef 2050 Plan, the first key challenge that it talks about is
on climate change. It talks explicitly about actions under the Paris Agreement as well as adaptation. It does that
for a couple of pages, laying out a number of the items that Senator Waters talked about with respect to the
emissions reduction fund, et cetera. I just want to make sure that that is very clear. It is covered in the existing
plan.

Senator WHISH-WILSON: It is covered in it, but you don't, as you have rightly pointed out, say whether
the Paris Agreement gets implemented. That is my point. Dr Reichelt, can I ask you again? You used the words,
'strongest possible actions' to prevent increased emissions. Do you believe that preventing the opening up of new
coal mines like the Carmichael mine, potentially the biggest coalmine in the world, is the 'strongest possible
action' on global warming?

Dr Reichelt: This is a very common question I get. As I am here representing the marine park authority that
manages the marine park, we have made very clear over several cycles of the outlook report where the risks lie,
including maps of impacts that run out to 2050 in the outlook reports. The specifics of the impacts of a mining
operation on the marine park is something we looked at through the lens of our legislation. Commenting on a
specific project is not something that our marine park authority has responsibility for. The specifics of the impacts
of a mining operation on the marine park are something we looked at through the lens of our legislation. To
comment on the specific project is not something that our marine park authority has responsibility for. What we
do—

CHAIR: Can I clarify? Are we now straying away from the Great Barrier Reef into outcome 2 or outcome 4?
Dr de Brouwer: Outcome 2.

Senator WHISH-WILSON: Dr Reichelt, my question was in relation to your opening statement. It could
have been a personal view, so feel free to clarify if it was a personal view. Strongest possible action on climate,
reducing emissions—could you clarify that for me, please? Do you personally believe that the opening-up of new
coalmines is counter to taking strong and sensible action?

CHAIR: I think asking personal opinions is definitely out of order.

Senator WHISH-WILSON: No, it is not out of order. I have sought advice from the secretariat. It is not out
of order. Dr Reichelt, you have already expressed a personal opinion here today. Could I ask you again: do you
believe that the opening-up of one of the world's biggest coalmines is counter to taking the strongest possible
actions on climate change?

Senator Birmingham: Dr Reichelt can, of course, respond to Senator Whish-Wilson, but I think the opening
statement was clear in relation to the importance of action on climate change. I am sure you understand, despite
the way in which you litigate the arguments here, that climate change is addressed as a global response. That is
why it is the work through the Paris Agreement that is important. It is the commitments that Australia gives via
the Paris Agreement that are important. It is the commitments that India gives via the Paris Agreement that are 
important. It is the agreements, collectively, that nations give and then how they give effect to those, and
Australia will meet its commitments in relation to the Paris Agreement. That is how we give strong effect to
action on climate change—meeting our commitments. But, in terms of Dr Reichelt's opening statement, he, of
course, can deal with the content of that.

Dr Reichelt: When I am speaking in the committee, I am giving my expert view as it relates to the agency that
I am here to represent in estimates. My personal views are ones that are coloured by expertise, based in
credibility, so I do not want to speculate. I do not want to have a feeling about something. I tend to talk in reality.
If I could just remind you—

Senator WHISH-WILSON: A number of other experts have been quite happy to provide personal opinions
on this issue.

CHAIR: Just to be clear: under standing order 16, you are not required to give a matter of opinion on
government policy.

Senator WHISH-WILSON: I am not asking for a matter of opinion. It seems that everybody here is keen for
Dr Reichelt not to answer this question. I think it is a fair enough question—

CHAIR: That is absolutely not true. Just because you do not necessarily like the way he is answering the
question does not mean that he is not—

Senator WHISH-WILSON: No, no. He has not answered the question, and I am entitled to a line of
questioning on very key parts of his statement.

CHAIR: Senator Whish-Wilson, please do not keep talking over me.

Senator WHISH-WILSON: I could say the same thing to you.

CHAIR: I have just said, as long as—

Senator WHISH-WILSON: I have got the questions.

CHAIR: Really, Senator Whish-Wilson, please. This is the first morning of estimates.

Senator WHISH-WILSON: I have got the questions.

CHAIR: You may ask a question, I am just clarifying standing order 16, privileges resolution, on what Dr
Reichelt is and is not obliged to do. As long as you are not asking him for a matter of opinion on government
policy, please continue.

Dr Reichelt: I understood the question, and the authority's expertise and value to the Australian public, and
the Great Barrier Reef, is that we do everything we can to protect the long-term future of the barrier reef. A very
clear statement, from the authority's point of view, is that climate change impacts on coral reefs are predicted to
worsen and critically affect the survival of reefs globally, without the strongest possible efforts to reduce
greenhouse gas emissions. I went further this morning and said that decisive action on reducing the levels of
carbon dioxide and other greenhouse gases in the atmosphere is essential if we want to ensure the survival of the
Great Barrier Reef and leave a legacy of environmental stewardship for future generations. That is my strongest
advice to the globe, to those who attend the Conference of the Parties and to our Australian public, and I stand by
these comments.

Senator WHISH-WILSON: We are at an important point in history, Dr Reichelt. The views of experts of
your eminence matter in this debate. I am giving you the chance to say whether you think that opening one of the
world's biggest coalmines is going to be counter to what you just told us then.

Dr Reichelt: I could repeat my answer.

Senator Birmingham: Dr Reichelt could repeat his answer. I could repeat my answer—that it comes back to
the global effort in relation to climate change, informed by the Paris Agreement, to which Australia takes its
responsibilities seriously. Australia has made strong commitments under the Paris Agreement, and we will ensure
those are honoured, irrespective of any particular developments that may take place in Australia. And of course
we will continue to urge other countries to make strong commitments under the Paris Agreement and to ensure
they are honoured.

CHAIR: I will come back if we have time, but we only have 10 more minutes. Senator Chisholm and Senator
Urquhart, how many more minutes do you need?

Senator CHISHOLM: I think I need five.

CHAIR: That is fine. You can go ahead for the next five minutes.

Senator CHISHOLM: I am interested in the trip that the foreign minister took with the diplomats and I have
a question to the department: were they aware that the trip was taking place?

Mr Oxley: Yes.

Senator CHISHOLM: What support did the department provide for the trip to take place?

Mr Oxley: The trip was managed and conducted by the Department of Foreign Affairs and Trade, with
support from the Great Barrier Reef Marine Park Authority. The department's small contribution was for a couple
of officers from the department, myself included, to join the trip and have the opportunity to see first-hand what
the diplomatic corps saw and to be available to talk about the Reef 2050 Plan should any of the diplomatic corps
want to understand the scope of the government's response.

Senator CHISHOLM: In terms of the costs for the trip, did the department bear any of that for the foreign
minister and diplomats that attended?

Mr Oxley: My understanding—and I will come back on notice if I am wrong—is that the only costs incurred
by the department were the costs of our two officers travelling and participating.

Senator CHISHOLM: What about for Dr Reichelt? Did the Great Barrier Reef Marine Park Authority incur a
cost for the trip?

Dr Reichelt: No, other than my staff attending from Townsville up to Cairns for the day.

Senator CHISHOLM: In terms of the minister's comments that were made to the media—that the reef would
be able to survive and thrive and that there was no chance that they would be given an 'in danger' determination
on the World Heritage register—did the department provide a briefing to the minister before those sorts of
comments were made?

Mr Oxley: I have not seen those comments, so I do not know whether it is an accurate attribution or not. The
department provided briefing material to the Department of Foreign Affairs and Trade, and it provided a briefing
to the Minister for Foreign Affairs, and, in advance of the trip, both myself and Dr Reichelt had an opportunity to
spend some time with the foreign minister, to brief her on what she could expect to see when she took the
diplomatic corps to the Great Barrier Reef and to bring her up to date with the Reef 2050 Plan.

Senator CHISHOLM: I have those comments in front of me. They ran in the Cairns Post, which is the local
paper. The briefing notes that were provided to her—would you be able to provide them to us so that we could see
what was provided before she made such comments?

Mr Oxley: I cannot give you briefing that was provided by the Department of Foreign Affairs and Trade to
the foreign minister. You will need to take that up in their estimates committee hearing, I would suggest.
Senator CHISHOLM: But you provided the briefing notes, didn't you?

Mr Oxley: We provided comments on briefing that the Department of Foreign Affairs and Trade prepared for
the foreign minister. That would be a normal process of interchange between departments. The only direct
contribution or engagement with the minister was a face-to-face discussion.

Senator CHISHOLM: Would the contributions that the department made to those briefings be able to be
made available to us?

Mr Oxley: I will take that on notice, Senator.

Senator CHISHOLM: In terms of the budget, we talked about land based pollution, sediment control, crown
of thorns eradication and research. Can the department specify the amount of funding and the source of funding
for those actions for 2017-18 and the forward estimates?

Ms Parry: Senator, which actions are you referring to?

Senator CHISHOLM: To the crown of thorns, land based pollution and sediment control, fishing regulation
and research.

Ms Parry: I can break down specifically around crown of thorns, but I can give you an overall budget picture
of our various programs and roughly where they are directed towards. In terms of the Reef 2050 implementation,
supporting the implementation of the Reef 2050 Plan our budget from 2016-17 out to 2021-22 is $94.777 million.
The Reef Trust over its lifetime from 2014-15 out to 2021-22 is $210 million. That is the Australian government
contribution and does not take account of any external funding that goes into the Reef Trust. The reef program
from 2014 to 2017-18 is $82.75 million. We have other NHT funding of $6.493 million from 2014-15 to 2017-18.
Similarly, from 2014-15 to 2017-18 there is an additional $27 million from the Biodiversity Fund which is
directed towards the Great Barrier Reef Foundation. We also have the National Environmental Science Program
running from 2014-15 to 2020-21, which is $31.98 million. That makes a total departmental expenditure of $456 
million from 2014 to 2020-21. I should also point out that we spend as a portion of that, as I indicated earlier,
$18.8 million crown-of-thorns starfish control.

That is not all of the Australian government expenditure that goes towards protection of the reef. That was
outlined in significant detail in the investment framework that was released in December 2016, which showed that
the Australian government is spending $716 million over five years to support the implementation of the Reef
2050 Plan. That, combined with Queensland government investment, is $1.28 billion over five years to support
the implementation of the Reef 2050 Plan, which is part of the overall government commitment of $2 billion over
10 years. This funding also does not take into account the $1 billion Great Barrier Reef Fund that is administered
through the Clean Energy Finance Corporation, which is debt and equity financing available for reef investments,
nor does it take account of the additional $124 million announced in MYEFO that was referenced earlier.
Senator CHISHOLM: In terms of, for instance, the reef program, which I think was from 2014-15 to 2017-
18—

Ms Parry: Yes.

Senator CHISHOLM: that money runs out next year?

Ms Parry: That is right—in 2017-18.

Senator CHISHOLM: So there was nothing in the forward estimates for that program?

Ms Parry: That program is due to end in 2017-18 and the transition to the Reef Trust picks up investments
from 2017-18 onwards, as well as the implementation of the Reef 2050 Plan.

Senator CHISHOLM: Thank you.

Senator WHISH-WILSON: I have a few questions around your approval for the Queensland government's
Shark Control Program. I am not sure whether you are the best person to ask.

Mr Knudson: Senator, I think we would deal with that under outcome 1.5. It is an environmental regulation
question.

Senator WHISH-WILSON: Could you very briefly outline what the Great Barrier Reef Marine Park
Authority role was in approving the Queensland government's Shark Control Program? What evidence was
collected by you?

Mr Elliot: It is the same as any other permanent application that comes through to the Great Barrier Reef
Marine Park Authority. There is an amount of equipment under that program which is located within the Great
Barrier Reef Marine Park and under our zoning plan requires a permit from us. It is a joint permit because it also
exists in the Queensland coast marine park. As with many of our permits, it is therefore a joint permit. It goes
through a standard permit assessment process, which in this case included public consultation on the permit itself.
We did a permit assessment report. That was written by our staff and went to a delegate within the Great Barrier
Reef Marine Park Authority, who granted that permit. I cannot remember the exact date. I think it was about April
sometime. Consequently, since then, as can be done for most of the permits that we grant, there has been a request
for reconsideration of that permit, and that process is underway at the moment.

Senator WHISH-WILSON: I understand that you had a large number of submissions. Did you look at the
scientific evidence around any particular outcomes on granting approval for those permits?

Mr Elliot: Yes. We looked at the scientific literature. We also looked at the information provided from the
monitoring of that program, the catch, the catch of bycatch and the survival rate of bycatch and all the other
statistics that are associated with it.

Senator WHISH-WILSON: So drumlines and nets catch and kill a number of non-target and protected
species, don't they? That is a fairly well-established fact.

Mr Elliot: Nets more so, which is why there are no longer any nets in the equipment that is used in the marine
park. One of the things we liaised with Queensland over was the removal of the nets. There are only drumlines in
the program components that are within the Great Barrier Reef Marine Park. The survival rate of incidental
bycatch on drumlines is much higher than in nets.

Senator WHISH-WILSON: But they still do kill non-target and protected species, correct?
Mr Elliot: Some do not survive—that is correct.

Senator WHISH-WILSON: I will get to SMART drumlines in a second as to whether you have taken that
into consideration.

CHAIR: You have three minutes left.

Senator WHISH-WILSON: Three minutes? I better ask you: have you looked at SMART drumlines?

Mr Elliot: As part of the assessment process we went through, we have looked at the current state of some of
those technologies and also, as part of the process, requested information from Queensland as to what their
rationale for not including them in the application was. Based on the evidence provided—and bearing in mind, of
course, the application before us was not for SMART drumlines—we will make a decision on the application that
was before us.

Senator WHISH-WILSON: But you could have not granted it if you had chosen to insist on SMART
drumlines?

Mr Elliot: If we decided that the application that was before us had unacceptable impacts, then we could have,
and would have, refused it. I would also—

Unidentified speaker: What info did they provide?

Mr Elliot: I would not be able to go into the specifics of exactly what they did provide on it. That particular
matter is also one of the reasons which has been put forward for a request of our reconsideration, so we are
reviewing that at the moment. I would not want to pre-empt what we might do.

Senator WHISH-WILSON: Could you tell us how long that review process would take? And, aside from
that review process on request, when would the Queensland government come to you next to renew the permits?

Mr Elliot: The permit that we granted—the one under review at the moment—was for a 10-year permit, so
they would be required to seek a consideration in 10 years. I would also point out that, as part of that, one of the
conditions is for a scientific research program to investigate alternative—

Senator WHISH-WILSON: Other mitigation technologies?

Mr Elliot: arrangements as part of that. To answer your first question though, the marine park authority is
required to make that reconsideration decision by 6 June.

Senator WHISH-WILSON: Is it your expectation that the Great Barrier Reef Marine Park Authority will
make a commitment at some stage to phasing out all lethal shark mitigation—I would not even call them
technologies..

Mr Elliot: I would not be able to speculate on that. What would be required is for there to be sufficient
confidence in non-lethal methods to provide the required outcomes.

Senator WHISH-WILSON: There is no scientific proof—certainly in the literature that I have been able to
find; and the committee will be coming to Queensland and looking at this issue—that lethal measures like
drumlines or nets protect human life. Is that true? Do you agree with that?

Mr Elliot: Did you want to say something?

Dr Reichelt: No, I would have to go back to the actual arguments in the submission.

Mr Elliot: I would probably have to go back to the assessment report that underpinned our original granting of
the permit. But it is, certainly, a topic of some debate as to the effectiveness of shark control programs in general.
On the Queensland coast, that particular shark control program has been around since 1962. It was brought in at
the time because of an increase in shark attacks, or perceived shark attacks. Certainly, we have seen in other
jurisdictions around Australia that there has been an increase in shark attacks recently. Now, there have not been
shark attacks along the Queensland coast. So I do not think that you would be able to say that it is not effective or
that it is effective because of the fact that it has been going for so long. But, certainly, the outcome is being
achieved of—

Senator WHISH-WILSON: There is certainly no scientific evidence to prove that it is effective in making
you safe in the ocean. In a number of those drum line locations, there have been no recorded shark incidents
either. I am just interested in what evidence you based your approval of those permits on.

Mr Elliot: We do have a statement of reasons which outlines our rationale for that original decision. We do
have an assessment report that went to the delegate to inform that. I am not sure if that is published on our website
or not, but certainly we can provide that.

CHAIR: I think Dr Reichelt wanted to clarify something.

Dr Reichelt: When this line of questions is finished, I want to just quickly update the committee on the
meeting. Sorry, Chair.

CHAIR: Thank you.

Senator WHISH-WILSON: Last question then, just to Mr Elliot: there was no statistical or scientific study
that you used as an evidentiary basis for those permits?

Mr Elliot: We referred to the scientific literature that does exist, and we certainly had the statistical analysis
of the catch rates of the program, the historical shark statistics that predated the program. It is also worth noting
that this program is limited to those beaches where there is significant human use, and it ends up being about 70.5
kilometres of beach across the entire 2,300 kilometres of the Great Barrier Reef coastline. I would have to take
that question on notice—

Senator WHISH-WILSON: On notice, yes. Could you also confirm that in 2016 alone, 531 sharks, including
endangered species such as the great white and the grey nurse, were killed, according to the data from the
Queensland government; and, since the program began, over 50,000 dolphins, dugongs, marine turtles, rays and
whales have also been killed by these lethal technologies.

Mr Elliot: There is one thing I would point out. First of all, that assessment report is on our website. So the
assessment report that underpinned our initial decision is available on the website, with all the references to the
scientific studies that we used et cetera. To answer your question, the statistics you are quoting are, I believe, for
the entire program across all of Queensland. There have been no great white sharks caught in the marine park, and
I believe the last time a grey nurse shark was caught was over a decade ago—the same for mortality. One of
figures I do have to mind is that, over a 10-year period in which there were, from memory, 25 turtles caught on
drum lines, 24 were released successfully, alive.

Senator WHISH-WILSON: Okay—

CHAIR: You did say this was on notice.

Senator WHISH-WILSON: If you could take it on notice, Mr Elliot. You have not caught the target species;
you have just said that. The great whites—

Mr Elliot: You mentioned them specifically just then.

Senator WHISH-WILSON: Yes, and grey nurses. But, also, how many bull sharks and tiger sharks have you
caught, and have there been any recorded fatalities of either of those two species? That would be useful to know.
Mr Elliot: That, I would have to take on notice. They will be in the assessment report, too.

Senator WHISH-WILSON: Okay. Thank you.

CHAIR: I understand Senator Duniam could round this off. You have a couple of questions.

Senator DUNIAM: I could, very briefly, thank you. Back to our friends, the crown-of-thorns starfish—I read
recently that we are using household vinegar to kill them? Is there any more information you have on that, and its
success?

Dr Reichelt: Yes, I can provide more information. The current bile salts approach was seen as expensive and
more complicated than it needed to be. It turns out that vinegar is a disruptor of the order systems of the starfish,
but I will have to check on its success. There has been a period of scientific assessment of it. What I need to check
is: Is it as effective? Is it a win in terms of cost-effectiveness? For me, it would be much better, in terms of the
ease of enabling other people to be delegated the role.

Senator DUNIAM: To head out with a boat—

Dr Reichelt: Well, we have to be careful, because being scratched by the starfish twice can lead to
anaphylaxis. It has very toxic skin. It is not something we encourage people to do lightly—
Senator DUNIAM: I am not volunteering!

Dr Reichelt: I did get a letter from Saudi Arabia yesterday, asking if we could explain how vinegar would
prevent coral bleaching—which it does not, of course; it is to do with starfish. They were going to douse the
whole reef in vinegar. I do think the simpler methods are better, and to upscale this activity we are going to need
something as simple and cheap as vinegar. I am fairly sure it will become adopted. There is a volume issue. I will
hand over to Bruce Elliot.

Mr Elliot: In terms of the use of bile salts versus vinegar, there is a logistics aspect to that as well. The bile
salts come in a powdered form; therefore, a drum of bile salts can be turned into a very large volume by mixing it
with sea water to then inject into the starfish. If you are doing a program such as the Australian AMPTO program,
bile salts are logistically better because they would have to have a huge barge behind them with thousands of
litres of vinegar, for example, to achieve the same outcome. But, on a smaller scale, with localised reefs being
looked after by tourism operators et cetera, vinegar becomes logistically far more viable.

Senator DUNIAM: So we await the results of the testing on this and we will see where it takes us. That is
very interesting. My apologies if this has been covered while I was out of the room: the International Coral Reef
Initiative and Australia's role in that. Can you give us an outline of that and what we are doing in sharing what we
have learned with other reef-managing nations?

Dr Reichelt: Yes. The International Coral Reef Initiative began with the Clinton administration and was led
by the US, Australia, Japan, UK and France, who all have a major coral reef interest, with the UK a bit smaller. It
has been running continuously since 1994 and Australia has had a lead role in that ever since. For nearly 10 years
Australia ran the Global Coral Reef Monitoring Network, which is a part of that program. It is a non-binding
agreement between the parties. It is a second track activity, in other words, between experts and managers of coral
reefs. It has gained interest from the NGO community. They meet annually and sometimes a second time for
special purposes. The chair rotates every two years and France currently has the chair. The French government
has taken a strong interest in it, as a means to alert the world to the risks of climate change, and we are strongly
supporting that. It was raised in a side event at the most recent conference of parties in Marrakech.
Australia is seen as a lead player in that initiative because of the 40 years of innovative developments in marine
protected area management and, more recently, in the coasts and catchments. The other element, apart from park
rangers and zoning plans, is that it is adaptable. We have a Reef Guardian program, which in Australia is the
education of school children, tourism operators and high standards. In the Caribbean, it is to do with the pollutants
that come from their piggeries on small islands. They have taken the concept of working with communities to
protect their reef systems. It has been very successful and it requires relatively few resources to operate.

Senator DUNIAM: That is pleasing to hear. Is that the only forum in which we do exchange knowledge with
other reef-managing nations or are there other ways we let them know what we have learnt?

Dr Reichelt: We have regular exchange with the National Oceanographic and Atmospheric Administration.

We have an MOU with NOAA, as it is called. It is their satellites which tell us the temperature. We, CSIRO, and
the Institute of Marine Science have worked together to come up with predictive heat maps for the ocean. Some
of those direct agency, interagency and intergovernmental links are very important. Another forum is the
International Society for Reef Studies, which has a four yearly conference. This turns out to be a major clearing
house and Australia is typically the highest representative there. That includes our universities as well as our
agencies. It is an area where Australia is recognised as world leader.

Mr Oxley: There is also collaboration through the World Heritage Marine Program. Dr Reichelt was involved
in a marine managers' conference last year in the Galapagos Islands. We also work with some of our nearer
neighbours through the Coral Triangle Initiative and through outreach in the Pacific.

Senator DUNIAM: Thank you very much.

CHAIR: Thank you. You had one point of clarification, Dr Reichelt.

Dr Reichelt: I can table the acceptance list for the conference electronically, and I will do that immediately
after this session. I want to let the committee know that we have international involvement, marine park rangers
from Queensland and ourselves, Biosecurity Australia, the Torres Strait, a number of traditional owners from
along the reef, AIMS, CSIRO, WWF, the Coral Reef Society, the University of Hawaii, the ARC centre of
excellence, James Cook University, five marine park tourism operators and peak bodies, the Tangaroa Blue
Foundation—we regard plastic pollution as very serious, with its local impacts on marine life—and some of the
other people with expertise in coastal water quality—the run-off issue. We have the Queensland Ports Association
involved, not QRC but the ports.

CHAIR: Was that it, Dr Reichelt?

Dr Reichelt: Yes, and we will give you the list of the names and affiliations.

CHAIR: Much appreciated, thank you for getting it back so promptly.

Ms Parry: I have a further clarification of an earlier item that Senator Chisholm asked about with respect to
departmental costs related to the diplomatic visit in Cairns. We also incur half of the cost for the Hon. Penelope
Wensley to attend in her capacity as chair of the reef advisory committee.

CHAIR: Thank you very much, Dr Reichelt and your team, for appearing here today. I now call the Director
of National Parks.

Director of National Parks

[12:31]

CHAIR: Welcome back, Ms Barnes. Would you like to make an opening statement?

Ms Barnes: No, thank you.

Senator Birmingham: Senator, before you start questions, I might on the record extend congratulations to Ms
Barnes, who I understand was recently awarded an ACT award for excellence in women's leadership. Ms Barnes
deserves praise and credit for her work in that regard.

CHAIR: Thank you very much, Minister. Ms Barnes, I am sure I speak on behalf of all the committee. We
extend our congratulations as well.

Ms Barnes: Thank you very much.

Senator URQUHART: I know that in 2015 the Abbott-Turnbull government sponsored a review into the
removal of about 127,000 square kilometres from the reserve, which is almost the size of my home state of
Tasmania. At that time, I understand that there was no justification given, but there has been a review. I have not
seen a response. Has there been a formal response to that review, given that it was a couple of years ago?

Ms Barnes: There has not been any removal of any size of the Commonwealth marine reserves. The outer
boundaries that were declared earlier are still in place. What the government did in 2014 was start a process to
look at what would happen with the management inside those reserves, and they have tasked to me to prepare new
management plans. The independent review that they commissioned is a wonderful input to my preparation of
management plans, but I have also sought input from a public consultation process and have had over 54,000
submissions to me on things I should take into consideration in preparing those management plans. Australia's
marine reserve network is the third largest in the world, behind the United States and France. The size of those
reserves has not changed in total. It is looking at how we do management within those reserves to deliver
environmental, social and economic outcomes.

Senator URQUHART: What are the time lines for the next steps? You said you have 54,000 submissions.

Ms Barnes: Working with a team at Parks Australia I have been looking through the submissions, looking
through the issues raised, looking through things I should take into account, and it is fair to say there are a mix of
issues. This is a national system, so there are lots of people interested in broad national interests and people
looking at local interests and I have been carefully considering their views.

In doing that I have do some work on what the potential impacts could be of various decisions, what the
benefits could be of various decisions, how we would work to have a marine park system that highly protects key
features but also looks at how we maintain jobs and also looks at how we make sure that if you have come into
Australia for a tourism experience you can have that in a well-managed marine park.

Senator URQUHART: In this year's budget, in the Environment and Energy portfolio statement under the
budgeted expenses for outcome 1, the expenditure laid out totals of only 28.3, but I understand the budget
commitment was $56.1 million. Where is the difference? Why is it spelled out like that?

Ms Barnes: The government's commitment of $56.1 million over four years is in the budget, it is just in
various places in the portfolio budget statement. Some of that money is in the Director of National Parks's
statement, some of it is in the Department of Environment and Energy's statement as administered funds, but the
total is still $56.1 million.

Senator URQUHART: So the total is there, it is just spread over different areas.

Ms Barnes: It is just spread out.

Senator URQUHART: Are you able to provide us—maybe not now—details of where that is highlighted?
Thanks.

Ms Barnes: Yes.

Mr Thompson: I might just clarify too that that expenditure is spread over a number of years. It is not a
single-year expenditure.

Ms Barnes: Yes, over four years.

Senator URQUHART: In completing the management arrangement, what approach is the government taking
to no-take the marine national park zones at a time when, globally, no-take zoning is increasing rapidly in the face
of incoming threats to the oceans, such as those experienced in Australia, including bleaching and mangrove and
kelp forest die-offs?

Ms Barnes: In providing advice to the government on the management plans I am looking at those studies. I
am also looking at all the signs and the economic and social impacts.

Senator URQUHART: Are you are looking at those in terms of providing advice to government?
Ms Barnes: Yes, and preparing the plans.

Senator URQUHART: Is the government looking to increase the no-take coverage from what was
established by Labor in 2012?

Ms Barnes: I have not made recommendations to the government yet.

Senator URQUHART: Will it be in that part of that plan when you finish putting all that together?

Ms Barnes: The draft plans that will go out for consultation will be my initial advice as to how I think we can
get those multiple outcomes.

Senator URQUHART: In relation to the plans being put on hold, are they being reviewed?

Ms Barnes: The original plans are put aside. The review looked at how you might put plans together and how
you could improve the original plans. But to get them finalised and through the statutory process, as Director of
National Parks I am now looking at all that input and developing management plans for 44 reserves.

Senator URQUHART: Does that mean that there are no protections in the marine reserves that have started?

Ms Barnes: There is certainly—we are managing those reserves in terms of putting together a science plan.
We are also managing those reserves from a compliance perspective working with other agencies. I might ask my
colleague, Mr Mundy, to run through some of the activities that are happening in those reserves, including the
work we are doing around marine debris and some of that work.

Mr Mundy: There are a range of activities going in reserves that are going on around the Commonwealth
marine reserve network at present, including science activities and particularly in the southeast network, which is
currently under active management and has management plans in place, as well as a series of 14 other reserves
around the country for which management arrangements existed pre-2012. Particularly in those locations, we
have active compliance and active science and management and permitting programs in place.

Senator URQUHART: Would it be fair to say that nothing has changed on the water or in the water, that
nothing is being protected that was in Labor's plans and that the government does not have any plans in place?

Ms Barnes: There are no management plans in place for where they have been taken away. But that does not
mean that we are not doing activities. So, for example—

Senator URQUHART: So there are activities, but—

Ms Barnes: the management plans are not there.

Senator URQUHART: So the management plans are not in place where they have been taken away?

Ms Barnes: No.

Senator URQUHART: I might just leave it at that at the moment, and I can come back in a minute if I need
to.

Senator SIEWERT: I want to go to the issue that I was pursuing last year in terms of Kakadu and the
management of the buffalo farm and some of those areas around there. I have the letter that you sent, correcting
your previous evidence, and I just wanted to pursue some of those issues.

Ms Barnes: Absolutely.

Senator SIEWERT: As I said, I have your letter. Can we now just confirm what the role of the Northern
Land Council in fact is, given the correction you gave.

Ms Barnes: I would be happy to update you on conversations I have had with the Northern Land Council
since then—

Senator SIEWERT: That would be great.

Ms Barnes: but I think in terms of details you should also speak to them directly.

Senator SIEWERT: To the NLC?

Ms Barnes: To the Northern Land Council.

Senator SIEWERT: Yes.

Ms Barnes: As the letter says and as the history shows, the buffalo farm has been in existence for a long
time—since 1990. It was at that stage run by the Gagudju Association, but they got into financial difficulty. At
that stage, at the end of the nineties, the Northern Land Council made a request to the then Director of National
Parks to take over the management of the buffalo farm. That was agreed to by the then Director of National Parks.
That was quite appropriate in that it is Aboriginal land, it was the traditional owners' request and the NLC is the
body that represents traditional owners on a number of matters, so the Director of National Parks, or the director 
of the park at that stage, approved that. I was talking to the Northern Land Council recently, and that
arrangement with the buffalo farm no longer seems to exist. In fact—

Senator SIEWERT: Between NLC and the—sorry to interrupt.

Ms Barnes: Their advice verbally to me was that that association is no longer there and that they do not have
that association with the buffalo farm. At the moment Mr Lindner is managing the buffalo farm, and I am
working under instruction from the board in line with the plan of management, which says that the current
management arrangements for the buffalo farm will remain in place. Once they change, the board would then
look at closing the farm.

Senator SIEWERT: Can you outline what the understanding is now about when they are going to change.
Ms Barnes: When—what? When the board—

Senator SIEWERT: The arrangements.

Ms Barnes: The arrangements are that Mr Lindner is continuing to manage the buffalo farm. He is providing
meat—both buffalo meat and, we understand, magpie geese meat—to traditional owners. The NLC is going to be
working through land claims in that area, and once they have finished those land claims they will be happy to talk
to traditional owners about what they would like to happen next with that particular part of the park.

Senator SIEWERT: What is your understanding of the time frame? I take on board what you said about
talking directly to the NLC, but what is your understanding about that time frame?

Ms Barnes: I could not answer on their behalf, nor could I answer on their behalf about how long it might
take to work through the land claim. That is very much an NLC with traditional owner process.

Senator SIEWERT: So Mr Lindner is now the manager of that area?

Ms Barnes: Certainly.

Senator SIEWERT: From your understanding, are there specific arrangements in place of the management
requirements for Mr Lindner over that land?

Ms Barnes: Now that I have had confirmation from the Northern Land Council that they do not feel that they
have that association with Mr Lindner, I am now going to be talking with Mr Lindner about his arrangements and
my expectations of his activity there. Bear in mind that he has been a longstanding manager of the buffalo farm.
Many traditional owners are very happy with the way he is running the buffalo farm. They are getting meat and
magpie geese. So I will talk to him now.

Senator SIEWERT: You would be aware that there are a group of people who are not happy, and are you
talking to them?

Ms Barnes: It is the NLC's role to tell me what the traditional owners' feelings are. The way the process works
is that the NLC works with the traditional owners around how they are feeling about things and consults with
them and provides me with advice. Until I get that advice, while the current arrangements stand and the
management plan and the board say to keep those arrangements until the current arrangements change, I will not
canvass traditional owners' views. But I will work with Mr Lindner.

Senator SIEWERT: Directly with Mr Lindner? Not via the NLC?

Ms Barnes: I will be working with Mr Lindner, and I am sure the NLC will be talking to traditional owners
about their views—both the people who support Mr Lindner in his operations and a number of those who would
have a different view.

Senator SIEWERT: In terms of public liability insurance, what is your understanding of the public liability
insurance responsibility of the buffalo farm at this stage?

Ms Barnes: These are the issues I have been talking to Mr Lindner about.

Senator SIEWERT: I am not trying to put words in your mouth, but does that mean you do not have an
understanding of where the public liability insurance responsibility stands at the moment?

Ms Barnes: As for Mr Lindner's public liability insurance, I need to talk to him about that and get
confirmation about what that is.

Senator SIEWERT: Is it expected that he should have public liability insurance?

Ms Barnes: I will need to talk to him and see if our liability would cover that area.

Senator SIEWERT: Under what circumstances would your liability cover that area?

Ms Barnes: I will have to take that on notice. I do not know the answer to that.


Senator SIEWERT: If you could take that on notice, that would be appreciated. So I can clearly understand:
you are going to be talking to him about that. You have to check the rules—

Ms Barnes: Now that I have had confirmation from the NLC that they no longer have an arrangement with
the buffalo farm—but I only learnt that at the end of last week—I now need to take the next steps to clarify what
arrangements need to be in place, and I will talk directly to Mr Lindner about that.

Senator SIEWERT: How long has it been that NLC has not had that direct arrangement? In effect you do not
know what—

Ms Barnes: I do not know. You would need to talk to them about that.

Senator SIEWERT: Shouldn't you also know, given that you are now having to liaise directly with Mr
Lindner about management of that area? In effect, there could have been a period of years where there was no
connection with the NLC or with National Parks.

Ms Barnes: I cannot answer that question because I do not know the answer to that question.
Senator SIEWERT: Has anybody in the authority been following this issue up?

Ms Barnes: I have been following it up with the NLC, and they have confirmed that they do not have an
arrangement with the buffalo farm. But they did not tell me how long it has been since they had that arrangement.
Senator SIEWERT: I just want to be clear: National Parks has not been pursuing this up until now, and it is
only now that you have been pursuing it?

Ms Barnes: Our staff liaise with Mr Lindner regularly and have discussions with him. But in terms of the
NLC's arrangement, I was not aware that that had changed until last week.

Senator SIEWERT: I understand what you are saying. I just need to be sure. I am just finding a little bit
strange that there seems to have been a period of time where the NLC did not have an arrangement and National
Parks was not aware of that and has not been talking to Mr Lindner.

Ms Barnes: We have been talking to Mr Lindner, as we would a range of people, on the park. But in terms of
his arrangements with the NLC, that is right.

Senator SIEWERT: Thank you.

Senator WHISH-WILSON: In relation to the question Senator Urquhart asked you, we were expecting that
the marine protected areas or reserves program would be released in the first quarter of 2017. Are there any
reasons for the delays that you can share with us? Are there budgetary constraints, or are there particular
stakeholders that have been holding up the process?

Ms Barnes: The budget is there. It is a very complex area. Making management plans for 36 per cent of the
Commonwealth waters—

Senator WHISH-WILSON: I understand.

Ms Barnes: is very complex. Making sure you put in place, as I say, arrangements that can deliver multiple
benefits and do not have unintended consequences has to be worked through very carefully.

Senator WHISH-WILSON: You may argue you could be there indefinitely on that basis as well. Are you
expecting, as on the website, that there will be a draft plan released in the next few months?
Ms Barnes: I would hope so, but there is economic and environmental information that needs to be worked
through. As I said, it is a national program, so it is a large body of work.

Senator WHISH-WILSON: I appreciate it is very complex. I was just interested in if there were any
particular stakeholders that were holding up the process.

Ms Barnes: No. It is that we are looking very carefully at alternatives and how else we might do things.

Senator WHISH-WILSON: I have a couple of quick questions on Operation Green Parrot on Norfolk Island.
You could perhaps take this on notice as we do not have a lot of time. Would you be able to give me the amounts
for the last few years and the forward estimates for the Norfolk Island National Park? Would you prefer to take
that on notice?

Ms Barnes: I will take that on notice.

Senator WHISH-WILSON: In particular, I wanted to know what amount of funding you are going to be able
to deliver on the actions in the management plan and how that relates to the line item on the green parrot. You can
take that on notice, too. We were pleased to see that Operation Green Parrot raised more than its target to help
with the relocation of the green parrot. It raised about $86,000. Was this crowdfunding actually done by the
national park?

Ms Barnes: It was a partnership with BirdLife Australia and a number of other groups. It was about raising
the funds so we could accelerate the work, but it was also about getting the constituency and having people
involved in the actual work. It was very successful.

Senator WHISH-WILSON: On the crowdfunding page you said:

By the time this campaign closes, park rangers will be preparing to move the first fledglings. But they will not be able to do it
without your help. Your donation is urgently needed now!

It is good to see that that was successful. Is this going to be a model for you guys going forward for these kinds of
projects?

Ms Barnes: On a case-by-case basis when we think it is a way that people can help accelerate some
conservation actions and where people want to be engaged. People want to be involved in these sorts of projects
and to hear about how they are going and to contribute—helping build aviaries, moving birds across and that sort
of thing. It is a good collaborative project. It was well received on the island. It also gave the island a lot of
publicity and people were coming over and supporting the island. So I think it was very successful.

Senator WHISH-WILSON: The creation of an insurance policy for the parrots was an important part of the
management plan. Do you also see it as perhaps a sign that some of these urgent tasks are underfunded in your
parks funding?

Ms Barnes: It is more about how we can accelerate some of these actions. We would have got there with the
funding, but with extra funding we could go a bit further a bit faster.

Senator WHISH-WILSON: Great, thanks. Please take those first two questions on notice.

Ms Barnes: Sure.

CHAIR: Just before we suspend for lunch, I have a couple of very quick questions. First of all, in this
committee we have had a number of discussions about yellow crazy ants. I am just wondering if you can give us
an update on the program on Christmas Island with the wasps.

Ms Barnes: Yellow crazy ants are a favourite topic of mine! Christmas Island yellow crazy ants are unique
and special.

CHAIR: Particularly crazy!

Ms Barnes: Yes. They have been there for quite a while, living in a sort of balance with the ecology. Then
just through an explosion of food supply the colony became a supercolony. After lots and lots of research and lots
and lots of looking at the different options, as you know, the scientists recommended that we import a microwasp
from Malaysia to rebalance the food supply and help rebalance the number of ants. So we brought over a small
number of wasps. We brought over 352 individual wasps to Christmas Island on 6 and 13 of December last year.
But they are breeding like wasps and we expect that by the middle of this year we will have more like 250,000
wasps. They have been breeding in a special wasp greenhouse, and then we have started to release them to small
areas where we can watch them and see what is happening. They have been released to four areas, and we will be
releasing them more progressively over the next year.

CHAIR: In those first releases, is there any sign that it is going to be successful?

Ms Barnes: It is too early to say, but certainly they are doing what they should be doing. As I said, they are
breeding. Then they will start to get into the food supply and hopefully disrupt the yellow crazy ants.

CHAIR: What is the ants' food supply? You said there has been an explosion in their food source.

Ms Barnes: The yellow lac. The wasp will lay its eggs in that food supply and disrupt that supply. The
reduction in the food supply will reduce, hopefully, the ant numbers. We will never eradicate them, but we think
this is a much better method than the ongoing use of chemicals.

Mr Thompson: Senator Moore asked earlier about conferences that the department has participated in in
relation to Sustainable Development Goals. Obviously in addition to those meetings where SDGs have been
discussed in international and domestic settings, there have been two major domestic conferences where the
department has participated. One was the Australian SDGs Summit in Sydney in September 2016 and the other
was the SDG Australia conference in Sydney during November 2016.

CHAIR: Thank you very much. We will make sure Senator Moore is aware of that. And thank you again for
getting back to us so promptly and for appearing here today.

Proceedings suspended from 12:55 to 13:45

Bureau of Meteorology

CHAIR: Dr Johnson, would you like to make an opening statement? 

Dr Johnson: No, thank you.

Senator CHISHOLM: I am interested whether any weather stations or sites have been decommissioned in
this term of government.

Dr Johnson: Weather stations or sites?

Senator CHISHOLM: Yes.

Dr Johnson: Certainly over the last couple of years we have been in the process of rationalising our footprint
in the organisation, and particularly in a number of our sites where we have previously had staff we are moving to
automate those. If you bear with me, I can talk to that in more detail. Since we last met, we have been in the
process of automating a number of stations, including Longreach, Mackay, Port Hedland and Williamtown; they
are in process. Over the last 12 months, we have also been going to automation in Weipa, Moree, Woomera,
Charleville, Mount Isa, Kalgoorlie, Cobar, Mount Gambier, Meekatharra and Halls Creek.

Senator CHISHOLM: In terms of these decisions, do you make announcements that these closures are taking
place?

Dr Johnson: Yes. These decisions are part of a broader strategy that the bureau announced some time ago
around our observational infrastructure. We have worked very closely with both staff and communities in which
those changes in those stations occur, just to make sure that that transition is as smooth as possible. We both
respect the interests of our staff and make sure that we have ongoing service continuity in those places.

Senator CHISHOLM: In terms of community consultation, can you give me a broad outline of what would
take place in terms of decisions made?

Dr Johnson: It would really depend on where we are talking about. As a general rule, we would engage key
local stakeholders. We work very closely, for example, with the local member of parliament if particular industry
or community sectors may be impacted. We worked with them, I know, along coastal Queensland, for example,
where we have had some changes in recent years. We work very closely with local government and emergency
services just to ensure that those transitions are as smooth as possible and that we can still provide an outstanding
service to the local community. In some cases, the automation investments we are making provide a superior
service to the local community than what we had when we had manual field observers.

Senator CHISHOLM: In terms of the stations that have been either decommissioned or are about to be, are
they ones that have been responsible for gathering information on flooding and providing, I suppose, information
that assists in providing flood warnings?

Dr Johnson: All observational infrastructure that we have contributes to our forecast and warning services. So
at the macro level, I could say that almost every piece of infrastructure that we have in the field contributes to our
forecasting capacity, so it is difficult to be specific. Any particular site would be a part of an enormous network.
For example, we have over 6½ thousand rainfall stations around the country. They all contribute every day to the
national forecast.

Senator CHISHOLM: I was specifically interested in Mackay because I have spent some time there recently.
There were areas that were affected by flooding as a result of tropical Cyclone Debbie that previously had not
been affected by flooding.

Dr Johnson: Sorry, can I just get you to clarify? What was affected by flooding?

Senator CHISHOLM: Some of the towns outlying from Mackay that were affected by flooding that had not
previously been. Has the decision around the station there had any impact on that?

Dr Johnson: I am not sure what places you are talking about that have been affected by flooding this time that
had not been previously. What I can say is that the changes in the observational infrastructure that we have made
in the Mackay region have not had any negative impact whatsoever on our capacity to issue outstanding forecasts
to the local community. As I said before, the replacement of assets that we are doing here, I think, actually
improves the skill of our forecasting and provides improved information to those local communities. But I would
be happy to take on notice if there is a particular set of communities that you feel are impacted negatively. I
would be happy to look into it.

Senator CHISHOLM: In terms of the budget saving from these changes, is that something that has been
identified?

Dr Johnson: Certainly the changes that we have made have delivered savings to the bureau. These savings are
absolutely critical in terms of ensuring that we can provide an ongoing service for the infrastructure that we do
choose to retain. 

Senator CHISHOLM: Are you able to identify what that is in terms of the amount of money?

Dr Johnson: I do not have that with me, I do not think, but I could certainly take that on notice.

Senator CHISHOLM: Yes. If a decision was made that we wanted to recommission a site for various
reasons, how easy is that to do?

Dr Johnson: I am not sure using the words 'easy' or 'hard' is the best way to describe it. If stakeholders wish to
talk to us about a particular issue in a particular region where they see a deficit in information, we would be
certainly happy to talk to them about the specific circumstances of what that perceived deficit is. Depending on
what the nature of that is would determine what the bureau's response would be. But I think it is important to
understand that we do have an observation system strategy. It is a six-year strategy. It commenced in 2014. A lot
of consultation nationally went on around that strategy. We are in the process of implementing it. Obviously, if
there were circumstances that caused us to deviate from that strategy, we would have to reassess what the overall
impact on our financial resources would be.

Senator CHISHOLM: Whilst I am tempted to filibuster until Senator Roberts gets here, I have no more
questions.

Dr Johnson: Thank you.

Senator DUNIAM: With regard to the forecasting systems and, in particular, Cyclone Debbie, are you able to
give us an outline of how the system kicks into gear and any improvements that have been made over recent years
to assist in dealing with these situations?

Dr Johnson: I would not mind just saying at the outset, before I respond specifically to that question, that
obviously Cyclone Debbie was a very significant weather event for our country. It was not just the cyclone itself
that caused significant impact in coastal Queensland. The flooding that resulted once the system had crossed the
coast caused very significant impacts, including the loss of nine lives. I think it serves us all well just to reflect on
the significance of that.

Senator DUNIAM: Absolutely.

Dr Johnson: I know my colleagues at the bureau feel that most deeply. I would also like to say that this was
quite a long-lived event. It placed very significant pressure on the bureau to perform over a sustained period. As
director, this is my first really serious nationally significant weather event. It was a privilege to see our people in
action. In my view, they did an outstanding job, as I said, under sustained pressure. The levels of forecast skill
and accuracy were very high. I think the bureau did the Australian community a great service over that period.
Certainly a number of leaders, including the police commissioner in Queensland and the Premier of Queensland,
have been on the public record complimenting the bureau on the quality of its services.

Senator DUNIAM: Absolutely.

Dr Johnson: It might be worth it, if I may, just to talk through the event—
Senator DUNIAM: Yes, absolutely.

Dr Johnson: because it was a very significant event. As you know, the cyclone made landfall on 28 March. It
was a very large and powerful category 4 system. It is interesting to know that the bureau issued warnings for 38
consecutive hours prior to and during the event. It is a very, very long period to be issuing consecutive warnings.
Debbie was a very large cyclone. The eye in particular was very large. The eye was in the vicinity of about 100
kilometres wide. Just to put that in context, the eye of severe tropical Cyclone Yasi—which some of you would
remember, which hit a little bit further north—was about 45 kilometres across. Severe tropical Cyclone Tracy,
which hit Darwin in 1974, had an eye of 12 kilometres. It just gives you some sense of the size of this system. It
was a very slow-moving system, which made it challenging from a forecast sense. Unfortunately for the people of
North Queensland, that slow movement meant that the impacts were experienced for a very long period of time. It
is also interesting to note that the largest wind gust that we recorded—I can remember that morning being in my
office in the Brisbane forecast centre and seeing that reading come in—was in excess of 260 kilometres an hour.
So there were very significant winds over a very sustained period.

The track and impact of the cyclone were entirely consistent with the forecast we issued days in advance.
Again, I think it is a great credit to our forecasters. We issued our first tropical cyclone watch on 24 March, so
that is four days before landfall. The first tropical cyclone warning was issued on 26 March. As I said, hourly
warnings were from 27 March all the way through to 29 March, when the system weakened. There were some
storm surges, as you would know. We had storm surges at Laguna Quays, just south of Proserpine, of up to 2.6
metres. That storm surge even extended as far south as Brisbane. The mouth of the Brisbane River had a 60-
centimetre storm surge, even though it was such a long way away. So it was a very, very significant event. 

After the cyclone crossed the coast, as you know, the system degenerated into a tropical rain depression.
Extremely high levels of rainfall were recorded. In the Mackay area, there were totals in excess of 1,000
millimetres. In one place, there was in excess of 1,300 millimetres. So there was very significant rainfall. That
rainfall extended down into the Fitzroy catchment. Again, I am sure you would have seen on the news media the
flooding in Rockhampton in the Fitzroy catchment and in the Connors and Isaac system, which are part of the
Fitzroy catchment. They had rainfall totals in excess of 1,000 millimetres. It was a one-in-500-years event in that
catchment.

I am also very proud of the way in which we predicted the flood peak in Rockhampton many days out. Our
forecast skill there was excellent. The system moved south and into south-east Queensland and northern New
South Wales. Again, in the Logan and Albert catchments, there was a one-in-20-years to one-in-50-years rainfall
event in the Logan and a one-in-100-years event in the Albert River. And then, in northern New South Wales, it
was a one-in-500-years to one-in-1,000-years rainfall event. A number of places in northern New South Wales
recorded in the vicinity of 700 millimetres during that event, which is a lot of rain. Records were broken in New
South Wales. There was a record flood in Murwillumbah, with 20,000 people evacuated. So it was a very
significant event. If it is all right with you, in addition in the outstanding performance of our meteorologists,
hydrologists and information systems specialists that kept all the bureau systems operating at full capacity—
Senator DUNIAM: I was going to ask about that, yes.

Dr Johnson: which is an extraordinary achievement over such a sustained period—I think our communication
with the community was extraordinary. We received over 600 media inquiries. Our senior people attended 11
media conferences at the state crisis control centre. We produced a number of videos on YouTube. We had over
half a million views on YouTube of our severe weather videos. We had nearly 5½ million Facebook accounts
reached with our warnings during the event, and about 1.15 billion views of the BOM video—

Senator DUNIAM: Billion?

Dr Johnson: Sorry, million. I should say million.

CHAIR: I was going to say that 'million' sounds a lot more—

Dr Johnson: My apologies. There were 1.145 million views on Facebook and lots and lots of tweets and
impressions and so on. So it was a full-court press from the bureau. It was a truly national effort. Although the
effort was largely directed out of our tropical cyclone forecasting centre in Brisbane, it drew upon staff from right
across the bureau—staff in Melbourne, staff in Hobart.

Senator DUNIAM: Yes.

Dr Johnson: Tasmanians contributed, Senator.

Senator DUNIAM: As they always do.

Dr Johnson: From your state. They contributed to the event. I am just enormously proud of our people, who
performed at the highest level when our nation needed us to perform.

CHAIR: I think it is safe to say that I speak on behalf of the entire committee, Dr Johnson, when I formally
acknowledge and thank your staff for the amazing work that they did. Quite often we think about emergency
services and other organisations but not actually your staff, who are clearly critically important throughout the
process. So I hope you do not mind passing on the committee's thanks.

Dr Johnson: Thanks, Chair. I know our staff would really appreciate that. It often is a thankless task. Usually
we get most of our feedback when we get it wrong. All feedback is gratefully received; it is how we improve as
an organisation. But I know for a lot of our people who have a massive passion and commitment to serving their
country, that sort of feedback is hugely valuable, so thank you.

CHAIR: Look, it is service to the country, particularly in the times when we need it the most, so we are very
sincere in our thanks.

Dr Johnson: Thank you.

CHAIR: On that, again, post these incidents, we think about the police, our military personnel and all the
normal first responders. I am wondering whether these sort of events place not just physical pressure; obviously
that is on your staff out in the field. I imagine that it would be quite psychologically challenging, knowing that
they have to get this right.

Dr Johnson: Correct.

CHAIR: Can you talk a bit about the impact of these events on your staff? 

Dr Johnson: Yes, thanks, Senator. It is a very perceptive question because it is a real watch point for me, as
chief executive. This last event was a very long and sustained event. My colleague Mr Webb will probably be
able to confirm this. Although Cyclone Yasi was a very large and intense cyclone, the actual period of intensity
for us was a matter of a couple of days, I think. This event went on for the best part of seven days and affected a
very large number of populated centres along the Queensland coast and into northern New South Wales. So our
staff do feel, I think—it is the flipside of their passion and their commitment—a deep sense of responsibility to
get it as right as we can and to provide that information as soon as we can to help our colleagues in the emergency
services community make the best possible decisions that they can, which often are also very difficult, risk-based
decisions. So it is a significant watch point for us. It is something we are working really hard on in the
organisation. It is not just the physical wellbeing of our people but their psychological wellbeing as well. For me,
as a new director in the organisation, the thing that has impressed me about the posture and psychology in the
bureau is it is something that is spoken about openly in the organisation. It is a healthy sign for me. Have we got
further to go? Absolutely. Have we injured some of our folks in the past? Yes. Are we learning and getting better?
Definitely. So I am pleased you asked because I know, probably on that Sunday after the flooding had sort of
come to a closer peak at Rockhampton but was still playing out in New South Wales, we were stretched to the
absolute limits, I think, physically and mentally.

CHAIR: So how did you manage that? This is my last question on this.

Dr Johnson: Yes, sure.

CHAIR: How did you manage it, then? Obviously, working people 12 to 24 hours for one day or two days is
hard but achievable. But when you have got limited resources of specialists, how do you maintain seven days?

Dr Johnson: There are a few things we do. Again, I invite my colleague, Mr Webb, who heads our national
forecasting service, to speak to this if I do not fully answer your question. We achieve it a few different ways.
Firstly, as you would expect, we would have a reasonable line of sight that something is coming and so we are
able to sit down with our teams and think about how we mobilise our manpower in expectation of the event that
comes. That happens not just at the local level. For example, with Cyclone Debbie, in certainly the initial phases
of it and the flooding that was in Queensland, it was through our Brisbane office. As the system moved into New
South Wales, it was through our Sydney office. So a lot of work goes on at that local level.

What really impressed me about this event, which mitigates the risk that you have identified, is that we were
able to mobilise staff from around Australia. So, for example, when the flooding was occurring at Rockhampton,
we brought in specialists based in Adelaide but who had worked in Queensland during their careers, so they
understood the local conditions. They had deep familiarity with the local context and local emergency services
personnel.

What we are working on hard in the bureau is a capacity to serve nationally whenever times of crisis arise so
that we are able to stand down our people, particularly our senior people, and stand up new senior leadership as
the event unfolds. That, at its core, is the strategy we are using—to try to take a truly national and whole-ofenterprise
approach to how we deliver our services so that we can move in an agile way when crisis comes.

CHAIR: Unfortunately, I guess, with these events, as you said, you now have the learnings about how to so
that if it does happen again, you know enough in advance so you can actually mobilise that.

Dr Johnson: Sure. Well, every event is a learning experience, because no two events, I think, are the same.
We formally conduct post event reviews. We have certainly done that in the New South Wales team. I think the
Queensland one is done as well. We also participate in post-event reviews with our colleagues in the emergency
services space because, as you can imagine, we have a really close relationship with our emergency services
partners. We have people, for example, embedded in the state control centres. So we are part of not only just a
learning process within the bureau but part of the learning that goes on more generally around the emergency
services community. I am looking at my colleague to see whether I have missed anything in that response, in
terms of how we mitigate that risk around psychological health and wellbeing.

Mr Webb: Thanks, Dr Johnson. I also remember that we are at the back end of an injection of forecaster
resources over the last three to four years. There was a review into the Bureau of Meteorology after the 2010-11
floods that resulted in an injection of forecasters to allow us to surge to those extremes. As Dr Johnson says, even
though the big events will stretch any organisation and require a really structured way of responding, we are
certainly at the back end of that project, at the moment, with more forecasters in our forecasting chairs. It is
making the response a lot more ordered. On top of that, as Dr Johnson said, the structured approach to
competencies around the country that allow people to step in wherever they are and standardisation of all of our 
services means that we can more easily respond to the ebbs and flows. We are very mindful of the mental health
of all of our employees as well.

CHAIR: Absolutely. As I said, on behalf of us all, please pass on our thanks for the professionalism, the work
and the service that they have given, particularly during these unfortunate events. Senator Chisholm, I understand
you have another question?

Senator CHISHOLM: Yes. I had the opportunity to check the name of the town that I was talking about,
which is Eton, west of Mackay.

Dr Johnson: It is just up near the Pioneer Valley, yes.

Senator CHISHOLM: There is a dam involved there as well.

Dr Johnson: Correct.

Senator CHISHOLM: Kinchant Dam.

Dr Johnson: Kinchant is above Eton, yes. It is just to the southwest of Eton.

Senator CHISHOLM: Is there anything you would be able to impart on what happened there with regard to
this?

Dr Johnson: It is probably best to take it offline. But my recollection of the period was that given the amount
of rain in that part of the world the dam overtopped and there was some risk of that overtopping, in terms of
communities downstream. But it is certainly not within the bureau's responsibilities, in terms of the management
of Kinchant Dam. There was a very large amount of rainfall in that part of the catchment.

Senator CHISHOLM: The state has initiated an inquiry into it.

Dr Johnson: Correct.

Senator CHISHOLM: Will you be participating in it?

Dr Johnson: I would have to double-check. I would be surprised if we would not make some form of
submission or be asked to provide details on the meteorological phenomena surrounding it. It is quite routine
when these things happen.

Senator CHISHOLM: In terms of the warning systems, just broadly, you quoted some impressive figures. Is
there any analysis of how that was seen in impacted areas? It is one thing if someone in Sydney, out of interest,
goes, 'Let's check how bad it is.' What about in the cyclone affected or flood affected communities? Is there any
separate analysis of the impact of the warnings?

Dr Johnson: Yes, is the short answer. I might ask Mr Webb to elaborate.

Mr Webb: Yes. Even while the event was still happening, we were arranging with universities and research
bodies and Queensland fire and emergency services to do some social science research into the effectiveness of
warnings. Traditionally we look at the accuracy of warnings through lead time et cetera and whether we picked
the right strength. But the real benefit from a warning comes when you understand whether people took the
appropriate action at the right time. Like the total warning system is a partnership, we are working with post
partners at the moment. I believe that some of the surveys have been done. They will be analysed in the coming
months so we understand the lessons learnt and how we can better penetrate with the messaging and words we
use in different situations in our warnings and the way we communicate, through things like social media and our
warnings, just to make sure that we give people the best opportunity to respond.

With some of these major events, it is the worst of nature and there is sometimes never enough warning for
some communities. But we want to make sure that we give them the absolute best that they can possibly get,
given the science.

Senator CHISHOLM: And are you aware of any concerns or issues, at the moment?

Mr Webb: Apart from understanding. At the moment, the feedback that we have had is more around how the
information is flowing within some of these events. I do not know any of the specifics and detail about concerns.
There is always the timeliness of warnings. We need to understand the types of warnings that people are getting
and whether they just did not get a warning that was issued and/or it was a different type of warning they were
using, or they were using a local community warning rather than official warnings. But with those types of things,
particularly during what we traditionally call our quieter season—the cool season in most parts of the country—
we will go back and analyse them. But the key is around making sure that people, when they need the
information, can grab it. And that does not just come from a warning. That comes from people understanding
what can happen in their communities. 

Senator CHISHOLM: So my understanding of the specific case west of Mackay is that the phone lines and
internet were down so that no warnings got through to a lot of these people.

Mr Webb: I am not aware of that situation. But the strong messages that we would give, in terms of the
preseason preparations, would be make sure you have a diverse range of technology, and the transistor radios and
the—

Senator CHISHOLM: The ABC was also down.

Mr Webb: That is why we build such strong links with local radio stations, including the ABC, just to make
sure, because we know that the internet goes down—

Senator CHISHOLM: The ABC was also down at the same time.

Mr Webb: Okay.

Senator CHISHOLM: So they had no phone line, no internet and, for a period, the ABC was also down. So I
think there is a specific sort of instance in that Mackay region that was a problem.

Mr Webb: Okay.

CHAIR: Thank you. We are in the unusual position of having finished questions for the Bureau of
Meteorology. We do not have representatives from the Climate Change Authority here yet, so I will suspend the
hearing until we can get them in. Dr Johnson, thank you very much for coming here today.

Dr Johnson: It is a pleasure.

CHAIR: Again, congratulations.

Dr Johnson: Thank you.

Proceedings suspended from 2.13 to 2.24 pm

Climate Change Authority

CHAIR: The committee has resumed with officers from the Climate Change Authority. Welcome, Dr Craik.
My apologies; I know it does not happen very often that we are actually ahead of time. Thank you very much for
accommodating us just a little earlier than we anticipated. Would you like to make an opening statement?
Ms Thompson: No. Thank you.

Dr Craik: I think they are out of breath.

Senator URQUHART: I want to talk about the funding allocation. Looking at the Environment and Energy
portfolio budget papers, the Climate Change Authority has been allocated $1.465 million in the financial year
2017-18 and nothing after that. Is that correct?

Ms Thompson: Yes. That is correct.

Senator URQUHART: I gather that the government has said that it will consider funding for the authority on
a year-by-year basis. That is, I imagine, because government policy is to wind up the authority in the life of the
current parliament.

Ms Thompson: Yes.

Senator URQUHART: So that is a significant cut in funding compared to the previous year, which was
$3.529 million. Yet, as I understand it, the staffing level in both of those years is reported as staying the same—at
nine. Is that correct?

Ms Thompson: It is a little bit of a moving feast.

Senator URQUHART: I am sure it is. People do not want to hang around, I guess.

Ms Thompson: No. I do not think that is quite right. One of the issues to bear in mind is that when the
authority was set up in 2012, it had rather more functions than it has at the moment. So when we were originally
set up, part of our role was to advise on the caps for the carbon pricing mechanism. We also had a role doing the
regular renewable energy target reviews. Both of those functions have been removed as a result of legislative
changes since. So while we do still have some statutory review functions, and while from time to time we are
asked to do special reviews at the request of the minister or possibly the parliament, in fact, a number of our
existing functions are no longer afoot for us. So I think that is partially reflected in the staffing profile. Another
thing to bear in mind is that when we were based in Melbourne there was the need to maintain a dedicated
corporate area just for us. With the move to Canberra in September last year, we no longer need to maintain such
a big standalone corporate function because it is easier to access shared services from the department and other
providers. 

Senator URQUHART: So what is the number? Is it nine?

Ms Thompson: We do have nine as our ASL, yes.

Senator URQUHART: So that cut in funding is about 60 per cent. You said that some of the functions are
less that you do. I guess what I want to know is whether the Climate Change Authority has the resources to
perform its legislative task adequately.

Ms Thompson: We believe so. In addition to the funds you mentioned earlier, we also have capacity to, in
effect, draw on. It is similar to the carry forward that other departments have. Basically, that means that we retain
unspent funds from previous financial years and use them for the coming financial year. There is a process we
need to go through where we have to get permission for an operating loss from the Department of Finance and the
finance minister. In the past, we have been able to access those funds as the operating loss and use that as part of
our financial resources to keep going. I guess I should also say that while it is probably true that everyone might
wish for a bit more on the resourcing front we do think we have enough resources to meet our statutory
obligations and our other functions.

Senator URQUHART: Good. Again, the budget paper says that the CCA has several reviews in train or
about to commence. In particular, it states that the authority will complete, by 31 December 2017, the second
review into the Carbon Credits (Carbon Farming Initiative) Act 2011. The authority will also begin work on the
review of the National Greenhouse and Energy Reporting Act 2007, which is due to be completed by 31
December 2018. So, given that, will you have to go and ask for more funding, or will it be part of that review that
you will get to finalise that?

Ms Thompson: I think the nine ASL we have should be sufficient for that task. We are fortunate in being able
to build up an extremely good team in Canberra. We have some very high-calibre staff working with us. We also
have the ability to obtain further resources as needed. If we find we need some further resources for economic
analysis, we will be able to procure them as well. So I think we should be well-equipped to deal with the CFI
legislative review and the national greenhouse and energy reporting review that follows it in the second half of the
financial year. We will start it then.

Senator URQUHART: The CCA's task is to provide expert advice to the Australian government on climate
change mitigation initiatives, including through conducting regular and specifically commissioned reviews and
undertaking climate change research. Do you and your workforce believe that this is important work?
Dr Craik: Yes, we do. That is certainly how we treat it.

Senator URQUHART: Yes, absolutely. Since the CCA was established in 2011, the science of climate
change has progressed significantly—is that correct?

Dr Craik: Sorry—that the science of climate change has progressed significantly? Well, yes, there has
certainly been additional science. I point out that we are not a science agency.

Senator URQUHART: No. I understand that. But you would agree that the science has progressed?
Dr Craik: We accept the broad thrust of climate science. We accept also that there are people who have
different opinions.

Senator URQUHART: So would it be accurate to say that the predominant scientific view is that the likely
impacts of climate change are now greater than what we understood back in 2011?

Dr Craik: Well, based on the fact that we are not scientists and we take the view of science agencies, that
would seem to be the prevailing view, yes.

Senator URQUHART: In light of the fact that we do understand more about the impacts, do you believe the
mission of the CCA has become more or less important over the last six years?

Dr Craik: Well, I suspect it has always been very important.

Senator URQUHART: Has that grown with more understanding of the science?

Dr Craik: Well, I think probably it depends on the question that we are asked to look at. I guess since I have
been there, we have been asked to look at what policies we would recommend that Australia use to meet its Paris
targets. Now we have another review on foot that the government has given us. We are doing a self-generated one
as well. But our view would be that they are all important. I guess we treat them all as very important at the time.

Senator URQUHART: Again, according to the budget papers, the CCA will be wound up in this term of
parliament. Does this intention have an effect on staff morale? What is staff morale like?

Dr Craik: Well, I would say that staff morale, from my perspective, is actually very good. They are very
impressive. They are very committed. They are very professional. They work really hard. I have never heard them 
complain about it. I think they find the work professionally challenging and the topics challenging as well. No, I
do not think it has an effect on staff morale. The fact that we have been able to recruit really good people, I think,
is an indication of that.

Senator URQUHART: So what preparations, if any, are being made for the closure of the CCA?

Ms Thompson: For the CCA to be wound up, legislation would need to be passed through the parliament. We
have on foot a set of standing procedures that we would go through if and when that becomes imminent.
Senator URQUHART: Are you confident that you have the staffing capacity to fulfil the current workload,
Dr Craik? You indicated that you had been recruiting. Do you have, I guess, the capacity to fulfil the current
workload?

Dr Craik: The current workload? Yes, I think so.

Senator URQUHART: Given it is unlikely that the government will introduce let alone pass legislation to
totally abolish the CCA, is not allocating funding a failure to comply with the legislation that established the
authority?

Ms Thompson: I do not believe so. The question for me, as the accountable authority, is whether we have
enough resources to meet the needs of our work program, particularly our statutory obligations for legislative
reviews. There are some other requirements as well around the PGPA Act reporting and so forth. But we have
met all of those, and I believe we will continue to do so.

Senator URQUHART: Right. But you did say earlier, if I am correct, that to wind it up, it does require an act
of parliament?

Ms Thompson: That is right. The authority is established in legislation, so to wind us up, you would need to
pass legislation to do so.

Senator URQUHART: Thank you very much.

Senator RICE: I want to ask some questions about your land sector emissions issues paper. I understand that
it has recently been released. Tell me, first of all, about that paper and what the timing of the submissions to it are.

Dr Craik: I suppose, to go back a bit, we initiated it ourselves out of the Special Review 3. One topic came up
in our review of our policies for different sectors. When we looked at the issue of agriculture, we saw policies for
reducing emissions in agriculture. Under different policies, there are policies for natural resource management.
Really it is trying to say, 'Well, are there any ways that those two policies could be brought together to generate
two lots of benefits with one policy?' So right now, you pay people to plant trees to reduce emissions and you pay
people to plant trees for biodiversity, but it is a shame that you do not plant a bunch of biodiverse trees when you
are planting the trees to reduce emissions. There are other activities. For instance, if you own a cattle production
operation, if you feed the cattle different supplements or you feed them different things, they produce fewer
emissions. You can turn them off earlier. All those things improve your productivity, so you make your money.
They also reduce emissions. Trying to find ways to generate more than the emissions reducing benefit and bring
those policies together was the purpose of it. So we put out an issues paper. We called for submissions. We got
about, I think, 16 submissions. We are in the process of taking those submissions on board and revising our final
paper.

Senator RICE: Okay. So the paper has already been out for comment?
Dr Craik: That is correct.

Senator RICE: You have received submissions?

Dr Craik: That is correct, yes.

Senator RICE: When do you expect to have a final paper?

Dr Craik: Well, we are busy with another review at the moment that is due on 1 June, so we will be finalising
the natural resource management one after that. So probably July some time.

Senator RICE: In the paper, you say that you have excluded the commercial forestry sector from the scope of
the review. It says in the paper to allow a greater focus on the opportunities for improved outcomes in the
agricultural sector. Could you expand on why you have not included forestry?

Ms Thompson: As Dr Craik mentioned, we really wanted to focus on ways to achieve these co-benefits of
broader natural resource management and environmental outcomes—reducing emissions or storing carbon on the
land—and improving on-farm profitability. The great thing about some of these actions, we think, is that they can
achieve all of those objectives with a single policy measure. It may well be possible to do similar sorts of things 
with forestry, but those sorts of actions open up a broader set of issues and considerations. We really just wanted
to focus on, as the paper is called, action on the land with a particular view of on-farm activities.
Senator RICE: So it is very much an agricultural focus?

Ms Thompson: That is right.

Senator RICE: You note that forestry will be within scope for the authority's second carbon farming initiative
review, which is due to be completed by the end of the year. Do you believe that you are going to have the
capacity to consider forestry thoroughly by the end of the year, particularly looking at your funding over the
forward estimates? Is that capacity going to be there to do the forestry section justice?

Ms Thompson: Firstly, I should say that the actual precise scope of the CFI legislative review will be a matter
for the authority members, equivalent to our board, over the next period. That said, though, the Emissions
Reductions Fund, which is a mechanism that is supported by the CFI legislation, does cover forestry and action to
improve carbon sequestration from forestry. I believe there is a method that is being worked on at the moment
that either has been or will soon be made that covers some forestry activities. I think there is at least one other that
does. So certainly those sorts of activities are in scope for the Emissions Reduction Fund and would be a set of
issues that the authority could well look at in terms of its review of the CFI legislation.

Senator RICE: And what is the timing on that review, then?

Ms Thompson: That has to be finished by 31 December in this calendar year.

Senator RICE: Given that gross deforestation is the second largest emitter of carbon emissions, the second
largest land based emissions source, it seems to me that forestry operations should be a priority for the Climate
Change Authority to be considering.

Dr Craik: Well, certainly we will look at it along with everything else when we consider the scope of the
review. We will be putting out an issues paper, obviously, as we normally do in seeking submissions and
feedback.

Senator RICE: To clarify: you are going to be looking at some aspects of forestry or broadly forestry as part
of the review of the Emissions Reduction Fund?

Ms Thompson: Perhaps I have been a little imprecise and put us all a bit crooked. Certainly the emissions
fund covers forestry.

Senator RICE: To some extent.

Ms Thompson: So certainly that is in scope for the authority to look at. What I was just trying to draw a
distinction in was perhaps the precise aspects or emphases that the authority might want to consider when it does
decide what precisely it will be looking at.

Senator RICE: Because so far the emissions reductions side has operated completely disconnected with the
climate change implications of commercial forestry operations.

CHAIR: Is that a question or a statement?

Senator RICE: It is a statement. I am asking whether that is accurate.

CHAIR: Thank you.

Senator RICE: Certainly that is my assessment, and it has been one of the critiques of the Emissions
Reduction Fund. As you were saying, you are putting large amounts of money in to growing vegetation. At the
same time, the other hand of government is allowing the removal of some of the most carbon dense vegetation in
the world.

Ms Thompson: I think one of the issues that tends to always cause a bit of complexity in this regard is the
carbon accounting category of the land use change in forestry. Sometimes when we are looking at, as you say,
deforestation, we are actually thinking more about action to clear land rather than what may or may not happen
when one stops harvesting commercial forestry trees and not replanting them. They are actually a bit different in
terms of how the accounting treatment works. They can be different in terms of their policy implications as well.
But certainly the ERF does include methods that give credit for action to reduce deforestation and credit for action
to allow trees to regrow by things like taking the stock off the land. So those are in the ERF at the moment and, I
imagine, would be things that the authority would at least look at.

Senator RICE: Will your review look at the impact of commercial forestry operation on native forests? We
are not talking about growing new trees. It is the operations of commercial forestry, of intensive clear-felling in
particular, on carbon dense forests and looking at the emissions implications of that.
 
Ms Thompson: As I said earlier, I am reluctant to be too precise about what the authority will and will not
look at in that review. But certainly all those matters are in scope in terms of the legislation and what that covers.
Senator RICE: Okay. So it is a bit vague, then, at the moment.

Dr Craik: Well, the authority has not started to look at the precise detail of that review yet. We have another
one that is keeping us occupied at the moment.

Senator RICE: Did you consult with any ministers about whether forestry would be in scope for this land use
paper?

Dr Craik: No. We did not.

Senator RICE: The other element of land use emissions is enteric fermentation. Are you planning on
including that in your study of land sector emissions?

Ms Thompson: In terms of the NRM land paper, the current paper?

Senator RICE: Yes.

Ms Thompson: That is one of the things we explicitly look at in that paper, yes.

Senator RICE: Given that that is such a large contribution, it only had minimal addressing in the paper,
however. Looking at it, given how significant a source enteric fermentation is, it was required to be addressed
more substantially in the paper.

Ms Thompson: I am very happy to take that feedback on board, Senator.

Senator RICE: Thank you.

Senator WATERS: I have a few questions about the membership of the authority. You have had a few
resignations lately. Earlier this year, Professor John Quiggin, who is a Queenslander, resigned. Why was that?
Dr Craik: Well, I think he put out a public statement saying why he resigned. Rather than me speak for him, I
think he was concerned that there had not been an official response to Special Review 3.

Senator WATERS: Is that correct? Has there been a government response to Special Review 3?

Dr Craik: No, there has not been a response yet to Special Review 3.

Senator WATERS: Is the government legally required to respond?

Dr Craik: The government is legally required to respond.

Senator WATERS: Minister, why has the government not responded to that special review?

Senator Birmingham: Well, the government is considering its response in the context of the overall review of
climate change policies, Senator Waters.

Senator WATERS: Section 60 of the act requires a response within six months of the report. That was due by
February. So that is a breach of the act by the government. Would you accept that?

Senator Birmingham: I do not have a copy of the act in front of me, so I would have to take that on notice.

Senator WATERS: It does not send a very strong signal about the government's seriousness on climate
change if you are not complying with statutory deadlines to respond to important reports about climate policy.
Senator Birmingham: Well, the government has been quite transparent about the process it is undertaking in
relation to the review of climate change policies. That work is being actively progressed.

Senator WATERS: So is this the one that the minister described as a sitrep shortly after he took over the
portfolio, or has it been re-elevated back to a proper review, or am I confusing the use of that?

Senator Birmingham: Again, I am not aware of the particular quotes you are referring to.

Senator WATERS: Is this the long-awaited 2017 climate policy review that you are referring to?

Senator Birmingham: This is the government's review of climate change policy. It is to help inform the
meeting of the 2030 targets.

Senator WATERS: Okay. So is it back to a review rather than just a situational report, as the minister
initially described it when he assumed the portfolio?

Senator Birmingham: Senator Waters, I am not aware of the particular quotes that you are referring to.

Senator WATERS: They were widely reported and, sadly, not retracted. I will go back to the CCA.
Apologies if this will cover some of the ground that Senator Rice already covered. Can you outline for me what
your current reviews are on and where they are at?

Dr Craik: We have two reviews underway at the moment. The first one is the one that the government gave to
us six weeks ago, on 10 April, where they asked us to do a special review with the Australian Electricity Market
Commission for the purposes of providing advice on policies to enhance power, system security and reduce
electricity prices consistent with achieving Australia's emissions reductions targets under the Paris Agreement.
That report is due to the minister by 1 June. The other review we are undertaking is the natural resource
management and agriculture review. We are looking at the synergies between policies in those two sectors to see
if they can be brought together in any way. We hope to finalise that report, I suppose, shortly after we finalise the
review due on 1 June.

Senator WATERS: This financial year?

Dr Craik: I would probably say early next financial year.

Senator WATERS: The budget does not have any money allocated for the CCA going forward. Are you
anticipating being able to complete your work despite the complete absence of a budget allocation?
Dr Craik: We are. Earlier in this session, Ms Thompson outlined the budget that we do have and that we have
adequate resources.

Senator WATERS: My apologies for missing that.

Dr Craik: That is all right.

Senator WATERS: So you have some residual that can tide you over. Is that the nuts of it?

Ms Thompson: My understanding is that the government has said it will consider funding for the Climate
Change Authority on a year-by-year basis. You are quite right; we do often find ourselves with some money left
over at the end of the financial year. The arrangement has been that we are able to access those funds in
successive financial years.

Senator WATERS: Thank you. Minister, given that the government has tried on many an occasion to abolish
the CCA and has been blocked in the Senate, why is it that the budget still reflects a zero allocation for this
continuing body which has statutory obligations to perform?

Senator Birmingham: The government has provided $2.1 million in the 2017-18 budget to support the
authority. Funding arrangements for 2018-19 onwards will be subject to the usual budget processes.

Ms Evans: I might make a minor correction. There is a $1.456 million allocation. The briefing that we
provided to the minister was incorrect.

Senator WATERS: Okay. I have $1.465 million and $1.456 million in front of me in the budget papers.
Ms Evans: They are the correct numbers in the budget papers.

Senator WATERS: Thank you. Minister, can you say that again? You are saying that it is the normal process
and yet it is not in the forwards. It is normal that it would be in the forwards. I did not understand your answer.
Senator Birmingham: No. Funding for 2018-19 onwards will be subject to consideration in future budget
processes if required.

Senator WATERS: So is it still government policy to abolish the CCA?

Senator Birmingham: As was discussed with Senator Urquhart earlier, yes.

Senator WATERS: Hopefully, we can still resist you in that regard. They are all the questions that I have.

Senator XENOPHON: I want to ask the authority questions in relation to the special review to be undertaken
under section 59 of the Climate Change Authority Act, which you are familiar with. In respect of that, how is the
report progressing, because the report is due on 1 June? Do you anticipate that you may need a short extension to
complete that report?

Dr Craik: No. We are aiming to complete it by 1 June. Yes, it has been a tight time frame—there is no doubt
about that—but we are aiming to complete it by 1 June.

Senator XENOPHON: What independent modelling are you using in the context of that report?

Dr Craik: We have commissioned some modelling. We are looking at the work that has been done to date and
we are making some amendments and suggesting some revisions to the work. You might want to spell it out in
more detail.

Ms Thompson: Our terms of reference are for the work to be informed by independent modelling.
Fortunately, quite a bit had already been done. You may be aware that the authority commissioned Jacobs to do
some modelling for us in 2015. Similarly, the Australian Energy Market Commission also sought modelling from 
Frontier, I think, about the same time. So we have those two pieces of work that are already on foot that we can 
draw on. In addition, we have sought some further analysis from the Centre for International Economics. They are
working with us to provide some, as I say, further analysis looking at those previous modelling exercises. They
are also doing a bit more work for us that we have asked them to do. So we are hopeful that that will provide a bit
of a fresh eye on that modelling work and help everyone distil the most salient features.

Senator XENOPHON: Further to that, you are required to consider the views of the AEMC report. Have you
met with them in the context of this report?

Dr Craik: We have spoken to them many times.

Senator XENOPHON: Many times. I imagine you speak to them a lot.

Dr Craik: Telephone conversations many times; lots of telephone conversations and exchanges of drafts.
Ms Thompson: In addition, some representatives from the AEMC came to an authority members meeting,
aka board meeting, last month, I believe, and talked through some of the issues that we are dealing with in the
report.

Senator XENOPHON: Sure. Very briefly, what are the other inputs into the report? What are the factors you
consider in preparing this report for government?

Dr Craik: Well, the factors we consider are obviously the objective of the report and the criteria that the
authority is required to take account of in assessing policies in relation to climate change.

Senator XENOPHON: The objective is fairly clear—is it not?

Dr Craik: The objective is fairly clear. But we have to take account of the factors that we are supposed to
consider. The AEMC has a number of factors that they wish to consider. So we are taking all that into account.
We are taking into account the modelling. We are taking into account comparisons.

Senator XENOPHON: Sorry, I did not hear that. You are taking into account?

Dr Craik: We are taking into account the modelling that has been done historically, theirs and ours, and
putting it all together and having discussions about where we might end up.

Senator XENOPHON: So it has not gone to the board of the Climate Change Authority at this stage?

Dr Craik: Well, we had a preliminary discussion at our previous board meeting about what policies we
thought we should look at and investigate. But the final report has not gone to the board yet, no.

Senator XENOPHON: So that will have to go out between now and next week?

Dr Craik: Correct. That is right.

Senator XENOPHON: When is that board meeting scheduled?

Dr Craik: Well, we have a preliminary telephone hook-up tomorrow, but there may well be further telephone
hook-ups to discuss it, depending on where we get to.

Senator XENOPHON: I refer to the document entitled Policy options for Australia's electricity supply sector.

It was a special review research report of August 2016. This report is in favour of an emissions intensity scheme.
That is in the summary at page 11. It is referred to there first up as well as in other parts of the report. Is this one
of the inputs that you will be considering in the context of this report?

Dr Craik: All our historical work on this subject we will be taking into account in considering this report. The
AEMC is doing likewise plus the contemporary work. And it is the most contemporary work as well.

Senator XENOPHON: So is there anything new before you at this stage that would make you reconsider the
findings of the previous report—the August 2016 report? Is there any new material?

Dr Craik: We will be looking around to see what other factors we need to take into account and discussing
what other factors they might be.

Senator XENOPHON: What does 'looking around' mean?

Dr Craik: Well, it just means taking into account if there is other work that has been done that we should take
into account. Of course, the government has made a statement about the emissions intensity scheme.
Senator XENOPHON: Is that a factor you take into account or not?

Dr Craik: Well, obviously we broadly look at the environment. We broadly look at the work that is being
done. We have a discussion about policies. We assess them against the criteria that we have been given in our act
to assess them against. The AEMC uses the criteria that it uses. We come up with a conclusion.

Senator XENOPHON: I am actually quite disturbed by what you said, Dr Craik, in relation to that. I thought
you were constrained to look at the AEMC and the evidence in terms of the most efficient way to enhance power
and systems security and to reduce electricity prices consistent with achieving Australia's emission reduction
targets in the Paris Agreement. Are you saying that one of the inputs is the Australian government's views about
one scheme or another?

Dr Craik: No. I am saying that we look at the criteria we are obliged to look at under our act, and the AEMC
looks at it under their legislation. We have a discussion about the policies and we make a recommendation. We
also take into account what other work has been done and other information. I did say that. It is certainly
something that we are aware of. To pretend that it does not exist would seem to me to be—

Senator XENOPHON: But what is the relevance? You have been tasked to undertake this review. It says in
developing the advice you should draw upon existing analysis and review processes and be informed by
independent modelling.

Dr Craik: Which is exactly what we will do.

Senator XENOPHON: But you actually made reference there to the views of the government in respect of
this.

Dr Craik: I did.

Senator XENOPHON: I am not being disrespectful to the government. I am just saying that I would have
thought it would be outside the purview of the government, the opposition, the Greens or my views. These are not
relevant considerations for the purpose of the special review by the authority.

Dr Craik: Well, as I have said, in coming up with our final recommendations, we will take into account the
criteria that we are obliged to take in and that the AEMC takes into account. But, clearly, it is background to the
review the fact that, as we pointed out in our previous report, climate change policy has moved around a lot. It has
changed back and forward a lot. There have been lots of changes. I am saying that that is background to it all. But
when we come up with our recommendations, we will be taking into account those criteria that I have indicated.

Senator XENOPHON: But not the background. The fact is that climate change policy has moved around a
fair bit. I think the ALP was against an emissions intensity scheme, for instance, and now is in favour of it. There
have been various iterations. It is similar in the coalition.

Dr Craik: Certainly in our previous report, one of the things we tried to do was come up with a
recommendation that met our criteria but could be broadly acceptable.

Senator XENOPHON: But on the issue of whether it is broadly acceptable or not, you have been briefed to—
Dr Craik: I was saying our previous reports, Special Report 3.

Senator XENOPHON: My understanding is that the AEMC are very much in favour of an EIS as per the
December 2016 final report into the integration of energy and emission reduction policy. What happens if there is
a fundamental difference of opinion between the Climate Change Authority and the AEMC? How is that
resolved? What process is there to resolve that? How do you take that into account?

Dr Craik: We will work that out if we get to that point.

Senator XENOPHON: That is a very pragmatic answer. Energy intensive industries that I have spoken to tell
me that they are under extreme financial pressure because of the price of gas in particular.

Dr Craik: Correct.

Senator XENOPHON: Is that one of the factors that you consider in terms of the affordability of power
prices?

Dr Craik: That ability.

Ms Thompson: I will add a bit. We put out our report in August last year. Since then, a few things have
happened. One is that Hazelwood shut. So I do think it is probably incumbent on us to have a look at how the
context and the landscape and some very important developments in the national energy market have changed.
That is part of what we are trying to do.

Senator XENOPHON: Of course.

Ms Thompson: We are wanting to make sure that what we say is up to date and reflects where all these things
have got to.

Senator XENOPHON: In fact, there are some indicative price increases and businesses saying that their
contracts are out there for two, three and up to four times the price of gas, according to Innes Willox from the
Australian Industry Group this morning on radio, who was talking about that. I look forward to your report next
week. 

Ms Thompson: Thank you, Senator.

CHAIR: Thank you very much. Do any other senators have questions of the Climate Change Authority? If
not, Ms Thompson and Dr Craik, thank you very much for appearing here today. I now call officers from the
department in relation to program 1.1, sustainable management of natural resources and the environment.

Senator URQUHART: I started on Landcare earlier today and was sent to 1.1, so I will continue along that
line. Can you confirm whether the allocation of money apportioned to the environment portfolio from the $1.1
billion in the rollout of the National Landcare Program moneys referred to in the budget will stay the same as
previously, or will it rise or fall between now and 2023?

Ms Campbell: The government made an announcement in the budget of $1.1 billion over seven years for the
Natural Heritage Trust component of the National Landcare Program. In that budget measure, they said that
details were being worked through. Those details do include the range of proportions between the portfolios and
the priorities under that program, so those decisions are yet to be taken by government.

Senator URQUHART: Do you know when they will be taken? Do you know a timeframe?

Ms Campbell: We are working with government to do this as quickly as possible and we hope for an
announcement shortly.

Senator URQUHART: So given that there is a reduction in funding to the department, is the department
achieving the same or similar outcomes with the program?

Ms Campbell: Again, in terms of the future funding, the government has not set how the program will be
delivered and what the priorities are under that program. So it is very early. It is too early to answer that type of
question.

Senator URQUHART: Right. But would it be fair to say that given that there is less funding, some of the
outcomes will not be delivered—

Ms Campbell: The government—

Senator URQUHART: in various areas or one or two areas or whatever?
Ms Campbell: The government will need to decide how it wants to spend that money and what its priorities
are.

Senator URQUHART: So it will be prioritised. So there will be possibly some outcomes that just will not get
priority or funding?

Mr Knudson: I will add for a second. We talked this morning about the Reef Trust and how it has pulled in
matching funding in different ways. So we are looking across the board at ways to make sure that we have
maximum impact with whatever funding the government brings to the table with other jurisdictions, the private
sector et cetera and philanthropic contributions as well. So it is a broader question that we will be looking at.
Obviously, Landcare is one element. I want to draw the point that we have been looking at this and doing it
successfully in areas like the Reef Trust. We will look at that more broadly.

Dr de Brouwer: I want to follow up two points on, again, the efficiency or effectiveness of programs. We
have been putting a greater focus on priorities and what the impacts of those priorities are. So, for example, within
the threatened species strategy, that is one way of highlighting, again, 20 animal and 20 plant species that are
iconic, important and vulnerable. They are also umbrella species for dealing with habitat improvement. If you
focus on a species, it has consequential other positive benefits. Other species are captured by the activities done
for that species. That is a long way of saying that the prioritisation of activities is one way of dealing with that
issue, again, to ensure the effectiveness of policy. The other is around the information systems that we are
providing and a greater focus on having that information publicly available. As Mr Knudson talked about, that
information is a way to get private or philanthropic financial support, because people want a better understanding
of what is happening to the species or to a habitat and that information helps provide confidence for that private
activity. So those sorts of other focuses are one way of improving the effectiveness of policy.

Senator URQUHART: In terms of the philanthropic or the private funding that you are talking about, would
that be prioritised towards what your priorities are? If you gained private funding, would that then go into the
priorities that you have established? How would that work?

Dr de Brouwer: My colleagues can also talk about this. I think when you come down to, say, the threatened
species strategy, you find that the threatened species strategy was designed with other groups—with scientists and
other groups involved in on-ground activity. When we say it is the government's threatened species strategy, we
mean that it is a community based strategy. It is not unilaterally designed by government. It involves others. 

Senator URQUHART: I understand that, yes.

Ms Jonasson: I will invite the Threatened Species Commissioner, Gregory Andrews, to talk about a couple of
examples where he has both private and government funding to support exactly the outcomes that you are talking
about.

Senator URQUHART: I will put that on hold for a moment because I have some other questions on
Landcare. I will have some specific questions on threatened species a bit later; I do not want to spend time on that
right now. In general, is Australia's environment more or less resilient? How would you sum it up?

Dr de Brouwer: I think we go back to the State of the environment report and the discussion we had this
morning. There are aspects that are clearly strong—we went through them—with the environment. Then there are
areas of vulnerability ongoing. You cannot achieve outcomes with environmental management overnight. The
State of the environment report goes through those vulnerabilities, including around threatened species. I am
happy to go through it.

Senator URQUHART: No. That is fine. We did go through that this morning and I do not want to go through
it again. I guess I am interested in what, if any, improvements are a result of new policies from the current
government. I understand that things take time, but what things have been done to improve outcomes?

Ms Jonasson: If you are referring to the National Landcare Program, I guess we would have to come back to
you on that probably at a future estimates once the government has made their decisions about where they want to
allocate the funding. We could then probably give you more information on that, if that is okay.

Senator URQUHART: What is the rationale for the government spending less on the environment and the
sustainable management of natural resources over five years from 1 July 2018 when the threats to our
environment from climate change, invasives and habitat destruction are actually accelerating?

Dr de Brouwer: I think one of the statements in the budget papers is really that there is an overall need for
budget repair. This portfolio is also part of that budget repair. In a sense, that is one of the priorities of the
government. As I say, at the same time, we are trying to get greater effectiveness from our existing instruments,
both from regulation and from our programs and spending on the ground.

Ms Jonasson: I would also like to clarify that the total amount of funding that has been announced by the
government, the $1.1 billion, consists of the full allocation of funding that is in the current Natural Heritage Trust.
There have been no savings taken out of the Natural Heritage Trust or the allocation of that funding from 2016-17
as well as the $100 million that was announced in MYEFO at the end of last year. So that is where the $1.1 billion
comes from. There have been no savings taken from the Natural Heritage Trust in relation to this announcement
by the government.

Senator URQUHART: Thanks for that. Are you able to provide any more detailed funding by the National
Landcare Program subprogram for 2017-18 and over the forward estimates?

Ms Campbell: We can certainly provide the subprograms under the National Landcare Program for 2017-18.
The programs beyond the forward estimates have not been determined by government, so we are unable to do so
at this time.

Senator URQUHART: So you do not have them yet, but you do have them for 2017-18 and you can provide
them?

Ms Campbell: Yes. We could do them either now or on notice.

Senator URQUHART: Have you got them there now?

Ms Campbell: Yes.

Mr Dadswell: The subprograms under the current National Landcare Program include the Environmental
Stewardship Program, which in the 2017-18 year is $11 million; Sustainable Agriculture and Biosecurity
Incursion Management, which is appropriated to, and administered by, the department of agriculture and is $3.7
million; and pests and disease preparedness, which is appropriated to the Treasury and administered by the
Department of Agriculture and Water Resources and is $19.9 million. So that is in addition to the Natural
Heritage Trust that is appropriated to the department.

Senator URQUHART: Sorry?

Mr Dadswell: Which is appropriated to the department of the environment.

Senator URQUHART: Thank you. I do not have any more questions on Landcare, but I do have some on
other issues.

CHAIR: I will come back to you. 

Senator RICE: I want to begin asking some questions about the hooded plovers at Warrnambool. I asked
some questions at the last estimates session about the impact of horse training activities on the beach at
Warrnambool. This horse training activity on the beaches near Warrnambool occurs regularly. It is something that
has been deliberately identified by trainers as a strategy for training their horses. So in terms of a threatening
process, it is organised by trainers, is a regular activity and is repeating in a pattern. We have the racing club. The
trainers are bringing their horses down there and training them regularly on the beach. Would you agree that this
training is a series of connected events, as regular activities that are repeating, and that they are having a
cumulative impact on the plovers?

Mr Andrews: Thanks, Senator Rice. It is fair to say that that particular activity in that location is having an
impact on the population of that species at that location.

Senator RICE: And that that series of connected events and regular activities is having a cumulative impact?

Mr Andrews: The advice that I have had from Birdlife Australia and the scientists working on the hooded
plover is that for that specific population of the plovers, the activities, or what is called disturbance—horses,
people, motor vehicles et cetera—are having a negative impact on that location.

Senator RICE: Yes. And that that is because it is a series of connected events and it is having a cumulative
impact. Would you agree with that?

Mr Andrews: Look, I think that is probably—I do not mean this disrespectfully—a bit of a leading question. I
do not think it is fair for me to really make an assessment on that. I have not been there and seen them myself. If
you are asking about a species or a population of a species that I had direct knowledge on, I would be happy to
comment. But I think I would be guessing, Senator.

Mr Knudson: If this is about whether we should have a threatening process assessment or something along
those lines, we would be dealing with that under outcome 1.4. It is the difference between if it is an actual listing
of a species or a threat abatement plan or anything like that. That is in that area. Gregory Andrews, obviously here
as Threatened Species Commissioner, can certainly answer questions at that level. But if it is about a specific
legislative instrument, I think we can deal with that later in the session.

Senator RICE: I raised it in estimates last time. I wrote to the minister and got a response from the minister
that basically each individual horse training activity is an individual event, they are not having a cumulative
impact and so it is not a threatening action under the EPBC Act. I want to ask some further questions about that.
So is 1.4 the best place?

Mr Knudson: That is correct.

Senator RICE: I will move on, then.

Mr Knudson: Particularly as it has not happened yet, so we can absolutely count on it.

Dr de Brouwer: We will come back on the outcome statement. If there is still confusion around where these
things are, we will try to be clearer in that outcome statement for next time.

CHAIR: That would be helpful. You were not here this morning. We went through some changes. Are there
any others?

Senator RICE: Well, I have some other questions.

CHAIR: Before we get any further, then, perhaps with the secretary's indulgence, both of you could quickly
go through which issues. We will just make sure that we know which ones are for now and which ones are later.

Senator RICE: Okay. The other questions I have are about threatened species and our processes for
protecting threatened species, particularly in forest areas.

Mr Knudson: If it is a question with respect to the regional forestry agreements, that would be here.
Senator RICE: Okay. I will start, and let us see how we go.

CHAIR: And you have a question of clarification as well?

Senator WATERS: Thank you for your indulgence. I have some questions on underreporting under NPI of
the coal-fired power stations, particularly Bayswater. Is that 1.6 or 1.1?

Mr Knudson: That is 1.6.

Senator WATERS: And I have some questions about the gas funding in Budget Paper No. 2—the $80
million-odd for various different gas shenanigans. Is that 1.1?

Dr de Brouwer: Some of that is energy. Are you talking about the bioregional assessments?

Senator WATERS: Some of it is on that, yes. 

Dr de Brouwer: Then I think that is actually in 1.2 now.

Senator WATERS: And would the remainder be in 4.1?

Dr de Brouwer: In 4.1.

Senator WATERS: Thanks very much.

Senator RICE: I want to start by asking some questions in general about actions to maintain healthy
populations of our native animals and plants and how we avoid our animals and plants becoming more threatened
or at risk of extinction. I have a particular focus on animals, because otherwise my scope is going to be far too
broad. We have quite a few native animal species with considerably smaller populations now than we had 100
years ago. You would agree with that?

Mr Andrews: Yes. Of course, yes. We have lost 30 mammals since Europeans arrived. Feral cats were
involved in 28 of those 30 extinctions. One was the result of direct human action. That was the thylacine.
Senator RICE: And government policy is to avoid extinctions and to act to stop animals becoming more
endangered. Is that correct?

Mr Andrews: The threatened species strategy aims to do its best to avoid extinctions and recover species.
Senator RICE: What are the mechanisms for maintaining and improving the populations of animal species?

Mr Andrews: The key mechanisms for avoiding extinction and recovering our species are to control the
threats and to create safe havens and appropriate habitat. We control threats, for example, to the Leadbeater's
possum. Some of the threats are out of our control to some extent, such as climate change. The government has
policies, and the rest of the world is acting on climate change. But the level of climate change that is already
locked in will have an impact on the possum. For each species, the threats are different. It is really important to
look at it not only from a specific local population perspective but as a species as a whole. At the same time as
tackling the threats, species are affected by historical habitat degradation. For example, in the wheat-sheep belt,
200 years of activities have altered that landscape. That is how we have grown our food, so it is about a balance.
In that case, the main objective is to reconnect and focus on the quality of the habitat and reconnecting habitat.
One thing I learned very early on is it is not the total volume of habitat that is important; it is the quality and the
connection. Animals and plants can often survive in areas that are not utopian wildernesses. So they can survive
with forestry activities or with farming. There does not have to be a utopian wilderness perspective.

CHAIR: Do you mind if I ask a quick question to follow on from this? You are talking about feral cats. Can
you give us an update on the program to eradicate two million or 2½ million cats? Can you give us an update on
how that program is going in terms of threat?

Mr Andrews: Thank you, Chair. I just got back from Kangaroo Island on Friday, which will be the world's
biggest ever island to have feral cats eradicated. It is one of the priorities in the threatened species strategy. There
are more feral cats than people on Kangaroo Island. The farmers and the conservationists are united to get rid of
the feral cats because of their impact not only on the wildlife—dozens of species will benefit—but also on
agricultural productivity through sheep and chook farming. I think I can give you an update. I can also say that
Australia is very supportive of this issue. Last week, on my Facebook account, I directly engaged on this issue of
feral cats. One hundred and fifty-six thousand people commented or liked it. I worked out that roughly the
equivalent of one out of every 150 people in Australia actually engaged and were interested, and 99 per cent of
them were supportive. To date, to meet the targets of five feral cat free islands by 2020, 10 large predator proof
fenced areas on the mainland by 2020 and two million feral cats culled humanely, effectively and justifiably by
2020, the Australian government has mobilised $30 million. I can table a summary of particular projects. I will
just mention one of them.

CHAIR: Very briefly, Mr Andrews.

Mr Andrews: In addition to Kangaroo Island, at Newhaven sanctuary run by Australian Wildlife
Conservancy, which already has dozens of species and 27 mammals, we will be building the world's biggest
predator proof fenced area. So the strategy is not only establishing the largest ever island eradication of feral cats
but also the largest ever fenced area in the world to protect species from feral cats. I can table this document. It
has my scribbles on it, but my team will give me a fresh copy.

CHAIR: I am happy for you to table a fresh copy when one is available. Thank you.

Senator RICE: Are we returning to me?

CHAIR: Yes. Returning to you. 

Senator RICE: Thank you. Mr Andrews, you summarise the actions that need to be taken as controlling
threats, creating safe havens and maintaining or improving appropriate habitats. Is it right to say that all of these
actions for our threatened species are planned in a recovery plan? Is that a summary of the work that is done—the
recovery plans that are prepared?

Mr Andrews: Sorry to be not as responsive as you would like me to be, but recovery plans are actually the
responsibility of Mr Richardson. He is appearing under agenda item 1.4. He can give you the specific details on
recovery plans, including the progress of the Leadbeater's possum recovery plan.

Senator RICE: So where does the work that you oversee—the work that you were just explaining about feral
cats—fit into the work that is undertaken as part of a recovery plan? This is why I am totally confused.

Mr Andrews: My work intersects with recovery plans and other documents because my job is to lead the
implementation of the threatened species strategy, work with recovery teams and areas of the department that are
responsible for drafting recovery plans and work with the Threatened Species Scientific Committee. My job
overall is threefold. One is to raise awareness and support for threatened species. I mentioned at the last estimates
that I summarise that as competing with the Kardashians because more people know who they are than what a
numbat, quoll or quokka looks like. The second is to mobilise support from within government. I am proud to say
that we have mobilised $227 million for projects. The funding for those projects is consistent with the science.
Sometimes recovery plans do not have the most up-to-date science. I am reluctant to say that the funding for those
programs is to implement recovery plans because some recovery plans actually do not have the best science, and
the most latest and up-to-date science is what is required.

Senator RICE: So, if I want to ask some questions about how successful recovery plans have been and the
implementation of recovery plans, do I ask that of you now or in 1.3?

Mr Knudson: It is outcome 1.4. If you think about it, if it is anything that has to do with the EPBC Act and
the regulatory instruments underneath that—whether a species gets listed, what sort of recovery plan,
conservation advice; that is all related to the regulatory system—that is dealt with in 1.4. So, as Mr Andrews has
laid out, he has his three roles, which are more about raising awareness and mobilising funding and action across
the country. But the regulatory instruments are all dealt with in 1.4.

Senator RICE: In terms of where the funding for Mr Andrews's role comes from and the funding for recovery
plans, where is that reflected in the budget? Mr Andrews is in table 1.1.

Mr Knudson: That is correct.

Senator RICE: Which budget line is that?

Ms Jonasson: The office of threatened species is supported both from broad departmental funds as well as
funds out of the Natural Heritage Trust, where we can.

Senator RICE: Is there more detail about where it fits in? We have the Natural Heritage Trust.

Ms Jonasson: We use a range of funding sources to support the office. We get funding from—

Mr Andrews: Look, I could summarise by saying that the funding—I have provided the tables for this
funding before—comes from the Natural Heritage Trust suite of programs, being the 20 Million Trees program,
the National Landcare Program and the Green Army, and the National Environmental Science Program. For
example, we have for the first time ever a $30 million hub dedicated solely to threatened species recovery that
comes out of the National Environmental Science Program.

Senator RICE: Where can I find a table that articulates what money the government is spending on
threatened species?

Mr Andrews: I could give that to you right now. I have only got one copy, but I am very happy to table it. I
usually do, or I get asked for it at estimates.

Senator RICE: But that is your work. That is not the work that is undertaken under 1.4.

Mr Andrews: No. It is teamwork by the department as a whole, because of course we intersect. We do not
stop. Unfortunately, in this process, the agenda has to have something that is a bit more binary. But on a day-today
level fighting extinction, we are working across multiple issues and people and areas. So it is teamwork.

Mr Knudson: If you are looking for specifics on the funding related specifically to Mr Andrews's office and
its staff versus that of the regulatory system on threatened species—

Senator RICE: And the implementation of recovery plans and what is being spent on the implementation of
recovery plans. 

Mr Knudson: what I would suggest is that if the staff have that when we show up at 1.4, we would be
absolutely happy to provide it. Otherwise we can certainly take that on notice and give you those two breakouts
for Mr Andrews's office and the area that deals with the recovery plans.

Senator RICE: Thank you. I will talk to you again at 1.4, then.

Senator WILLIAMS: Mr Andrews, has Professor John Wamsley been involved at all in the program for the
reduction of feral cats? I know that it was his program many, many years ago. He used to wear a hat.
Mr Andrews: He did actually have a hat.

Senator WILLIAMS: A good man.

Mr Andrews: We have not been working directly with him, but his legacy is definitely something that we
work with and support. Australian Wildlife Conservancy is a nongovernment organisation that evolved from
Earth Sanctuaries, which I believe is what Mr Wamsley's organisation was called. So we are certainly working
and expanding his legacy.

Senator WILLIAMS: Good.

Mr Andrews: That legacy is the importance of fenced areas, particularly to protect our most vulnerable
species, such as the mala hare wallaby, which would be extinct if it was not for those fenced areas. I follow the
Wallabies rugby team. We cannot keep naming our sporting teams after our animals if we lose them to extinction.
So Mr Wamsley and Australian Wildlife Conservancy are really important players.

Senator WILLIAMS: I should know the answer to this question, but I do not: are there any foxes on
Kangaroo Island?

Mr Andrews: Luckily, there are no foxes on Kangaroo Island, which is what gives it one of the values to go
further and eradicate the feral cats. There are no foxes or rabbits there, so it already has strong biodiversity value.
But, because the foxes actually disrupt the cats, the cats are pretty much out of control. There is about one feral
cat, on average, to at least every two square kilometres—but often to one square kilometre. It is like a granny
square rug with each cat operating. Actually, last Friday, I cut open a feral cat with scientists and there were 28
animals in its stomach. We find lizards and native rodents, and even wallabies of up to five kilos get ripped in half
by feral cats.

Senator WILLIAMS: My wife is savage on cats, I can tell you.

Senator CHISHOLM: We will send Nancy to Kangaroo Island!

Senator WILLIAMS: She is a great shot when it comes to cats around the place.

Senator WHISH-WILSON: I have a couple of quick questions on the Tamar River dredging program. Mr
Costello, I understand that the funding runs out in 2016-17, which is the end of financial year 2017. Correct me if
it is calendar year. Could you give me a quick update on the process from here? Are there any feedback loops at
all in terms of the funding that has been provided?

Mr Costello: There was a $3 million commitment over three years. You are correct; it does finish at 30 June
this year. But there was a further commitment made for another $1.5 million for the next three years. So that is
being looked at as a second stage of the work that is being done. That new commitment has been incorporated
into the Launceston city deal that was announced in April, so there is some continuity of the work there.
Senator WHISH-WILSON: So that was announced when the Prime Minister came down to Launceston in
April?

Mr Costello: That is right.

Senator WHISH-WILSON: Were the outcomes of the dredging program assessed by the department or a
consultant?

Mr Costello: The outcomes from the dredging program were assessed by the program itself. They provided us
reports on that. We have talked about some of those before, with the finding being that the dredging is more
successful while there is good flow in the river. It makes sense. The sediment is disturbed to be carried
downstream with the flow. But in low flow conditions, it is much less effective. There were tracer studies and
other studies done to look at the movement of the silt around the estuary, because you have a tidal influence in
there as well. So there has been quite a lot of analysis of the effectiveness of the sediment raking.

Senator WHISH-WILSON: While there is flow in the river, like a flood or hydro releasing water into the
river specifically?

Mr Costello: Yes, certainly. So the major floods in June 2016 had a very large scouring impact. Obviously,
you cannot recreate that with releases. 

Senator WHISH-WILSON: Well, we hope we do not recreate that.

Mr Costello: So there was a very, very significant reduction in the silt load there. On the numbers I have, the
equivalent of 33,680 truckloads of sediment was removed by the floods.

Senator WHISH-WILSON: So why was the funding halved? It was $3 million to begin with. Why is it $1.5
million over the next three years?

Mr Costello: That was a decision of government. It was an election commitment made in 2016.

Senator WHISH-WILSON: Are you confident as a department, based on what you have seen so far, that
$1.5 million over three years will have any impact or have a measurable impact on the sediment?

Mr Costello: There are some new commitments, as part of the Launceston city deal, about how the funding is
organised. We are really looking for better governance of the overall estuary and more accountability for the
funding that is provided. So the Tasmanian government is now engaged more heavily. They have agreed to
establish a Tamar estuary taskforce with independent experts and local stakeholders involved. Their first job will
be a river health action plan to prioritise all of the potential investments that could be made to improve the water
quality in the Tamar.

Senator WHISH-WILSON: That was my last question on this subject. Will someone within the department
be on that taskforce?

Mr Costello: We will not be on the taskforce, no. It is a Tasmanian government and local council initiative.
But they will be providing a report to us with the prioritisation of the potential investments.

Senator WHISH-WILSON: I think you guys are the only ones putting in money at the moment.

Mr Costello: I am happy to report that the Tasmanian government, as part of the city deal, committed half a
million dollars on top of our $1.5 million, so that has taken that up to $2 million. So we were pleased with that
contribution and, in particular, the buy-in of the Tasmanian government into the issue as well. As you know, there
are many players. There is urban stormwater run-off. There is the combined sewage stormwater works. There is
catchment management higher up in the catchment, stabilising riverbanks. There is discharge from dairy. So there
are many, many factors that contribute to the condition.

Senator WHISH-WILSON: That is right.

Mr Costello: So we are trying to bring that together in a more coordinated way.

Senator WHISH-WILSON: Good. Thank you. That is all from me on this line of questions. I do have some
questions for the Threatened Species Commissioner.

CHAIR: Before you move, Mr Costello, Senator Chisholm, your last questions are to whom?

Senator CHISHOLM: Around the Indigenous protected areas and Indigenous Rangers program.

Ms Campbell: I can answer some of those questions.

CHAIR: Do you mind if we do this, because I have some questions? Senator Chisholm, we have time if you
want to ask your questions.

Senator CHISHOLM: Well, I have questions for the Threatened Species Commissioner as well.

CHAIR: If you want to do your questions now, we will get the commissioner back. Then we will finish up
with the commissioner.

Senator WHISH-WILSON: Could I suggest that I finish with the Threatened Species Commissioner? Then
we can go to Senator Chisholm and he can finish off both sets of his questions in that way. Otherwise you will
have to come back to me and cut them in half.

CHAIR: That is fine. Commissioner, you are back up.

Senator WHISH-WILSON: You are very popular today. My questions relate to an article published in the
Conversation by three scientists and a social media thread that you were involved with commenting on that
article. Just to give you the information—you are probably aware of it—it was called 'Government needs to fund
up billions, not millions, to save Australia's threatened species'. It was published on 21 March. You responded to
a tweet from one of the scientists, Dr Euan Ritchie from Deakin University, criticising him and his article about
the lack of funding for threatened species conservation. Mr Andrews, you said:
Shame to see erroneous figures used to support arguments in this paper. Consultation with my office before choosing to
publish could have ensured scientific accuracy.

I understand the scientist said that he would be happy to hear from you about any mistakes they made. Did you
get back to Dr Ritchie about what was erroneous in his article? 

Mr Andrews: Thank you for that question. Actually, I did not because I did not think that it was my
responsibility to correct erroneous science. I thought they should have contacted me. They invited me to make a
submission and correct the record, but I said to them, 'I am actually really busy fighting extinction. What if I spent
my time responding in detail to something that we could have avoided?' For example, they said that the
prospectus is really biased towards cute things and that it did not have enough plants. Well, actually, there are 333
species directly benefitting from the threatened species strategy in the prospectus, including species like the
bladderwort and various galaxias and things that are not cute.

Senator WHISH-WILSON: Mr Andrews, I might bring you back to my line of questioning, if you do not
mind. So you are happy to criticise them on social media, but you did not want to help clarify?

Mr Andrews: I responded to their criticism on social media. I do my best to respond to everybody who is not
rude. I then contacted them via email and we had a short discussion by email. They offered for me to write an
essay that could be considered to counter their point of view. I said, 'Look, thank you. I won't because I could
spend that time planning projects to fight extinction and do more grassroots things.' But I did say to them that I
would appreciate it if they consulted me, because it would be courteous to actually call me. I could have helped
them avoid wrong numbers in their science.

Senator WHISH-WILSON: I will be the devil's advocate here, Mr Andrews, and suggest that perhaps if you
had consulted them not on social media initially but gone straight to them, it might have been seen as a more
collaborative approach rather than criticising them publicly and then not backing up your criticisms with any
details or information.

Mr Andrews: Look, I disagree with that. They criticised the strategy and I replied to their criticism. I would
not initiate criticism of them, but I do think it is important to be correct and factual and on the record, and that is
what I did. So I was respectful and courteous. Actually, I have respectful and courteous relations with those
scientists and am happy to work with them and, indeed do; they are funded under a number of projects on the
threatened species strategy.

Senator WHISH-WILSON: This is where I will bring in this question. There was a tweet that you deleted as
well, which I am happy to read to the committee; it was quite critical. In fact, it was almost bitchy, if you do not
mind me using that word. You copied the environment minister into that tweet as well. Given they are getting
funding from the government, do you not think that is kind of threatening behaviour to scientists whose
reputations and funding depend on their credibility and ongoing funding sources?

Mr Andrews: Last year, I reached millions of people on social media. I do my best at all times to do that
respectfully. I have not had any complaints from those scientists that they were concerned about how I engaged
with them on social media.

Mr Knudson: I think your point nonetheless is absolutely valid. We always want to have constructive and
meaningful engagement with the full range of opinions in society. We will take this on board in terms of that, yes,
and we will continue to engage and note what you have raised here.

Senator WHISH-WILSON: Sure. I note the terms of reference of the Threatened Species Commissioner.

One of your roles, Mr Andrews, is to lead efforts to report on outcomes of conservation activities for prior
threatened species, including the effectiveness of specific investments and achievements. So I understand why
you would respond. That is perfectly understandable from my point of view. One of your other roles is to work
collaboratively with scientists; that is next. I think in this case those two things are contradictory. We have been
contacted by the scientists, who felt very intimidated by this particular exchange. Finally, perhaps you could—
you can take it on notice—let the committee know what was erroneous about that article. I understand you still
have not communicated that.

Mr Andrews: I would be delighted to say what is erroneous about it. It made an assertion—I cannot
remember the exact number—that only 20 or 17 plants were being saved. That is the first. They actually said—
and I accepted—that the projects were good projects.

CHAIR: So, Mr Andrews, you did have the offer to take that question on notice.
Mr Andrews: Sorry.

CHAIR: You have already corrected yourself once, if not twice, in this answer. If you are not quite sure of all
the facts, it might be better to be accurate and take it on notice so that we do not have to go back and clarify it
later. 

Mr Knudson: Again, I think we want to get on to a more positive footing going forward. I think getting that
written out and on the record formally and communicated with the scientists, I think, is important for setting some
right paths, so I agree with the Chair's comment.

Senator WHISH-WILSON: That is good to hear, Mr Knudson, because obviously we do not want the
commissioner to be seen to be in any way censoring scientists, who I think are raising a pretty important point
that conservation funding is underfunded by global standards. A lot of the data in that article related to
government funding. It was not in any way having a go at Mr Andrews or the work he is doing. It is actually the
taxpayer and the government putting enough into these outcomes. We would all like to see more funding. I do not
think it is an unreasonable thing to be raising.

Senator CHISHOLM: I might just ask about the Threatened Species Commissioner, since we are on the
topic, before I move on. I am looking for a specific example or examples of improvements that have been made
that the government can identify through the work of the Threatened Species Commissioner.
Mr Andrews: Thank you, Senator. So you are asking for specific examples of species—

Senator CHISHOLM: Yes.

Mr Andrews: that have had improvements. I actually made some notes on that. The helmeted honeyeater in
Victoria is an example of a species under the strategy that has an improved trajectory. So is the Norfolk Island
green parrot. That is a good one, because we have had almost a tenfold increase in the global population of that
species. We have now leveraged that with private sector support and investment through a crowdfunding initiative
to extend on the safe haven concept that I was talking about before. Rock wallabies in Kalbarri National Park are
improving thanks to the baiting of feral cats there. Eastern quolls and eastern bettongs, which were formerly
extinct on the mainland, are actually now back on the mainland and thriving here in the ACT in the predator
proofed fenced area at Mulligans Flat, which is part of the strategy and is being expanded. Bilbies are benefitting.
Right at the moment, over the next few months, will be the first ever bilby blitz to get a national survey of bilbies
across Indigenous communities and Indigenous managed and owned land. Species such as the silver daisy and the
magenta lilly pilly have upward trajectories as a result of the threatened species strategy.

Senator CHISHOLM: Thanks. Just in relation to the Indigenous protected areas, I want to confirm that
following the closing down of the Green Army program, money was allocated from the Green Army program to
Indigenous protected areas. Is that correct?

Mr Dadswell: As a result of the savings measures from the Green Army program at the mid-year economic
forecast, an additional $100 million over four years was provided for the National Landcare Program. Part of the
government's announcement on that program included $15 million for new Indigenous protected areas. There are
other Indigenous protected areas that are funded under the Natural Heritage Trust.

Senator CHISHOLM: What I am trying to establish is the total funding for Indigenous protected areas over
the next five years and having that broken down into existing Indigenous protected areas, ones under formal
consultation and then new Indigenous protected areas.

Ms Jonasson: What we might need to clarify here is that the Department of the Prime Minister and Cabinet
are responsible for the administration and management of Indigenous protected area networks these days. What
we can do, because some of the funding is still sourced from this portfolio, is give you a breakdown of the
funding. Some of the detail of your question may have to be put to the Department of the Prime Minister and
Cabinet.

Senator CHISHOLM: Sure.

Ms Campbell: Yes. The Indigenous protected areas funding comes out of the National Landcare Program. In
the four years to 2017-18, we have spent $64.681 million on the Indigenous Protected Areas program. Those are,
as you flagged, for the consultation and ultimate declaration of Indigenous protected areas. We have, I think, 72
Indigenous protected areas around the country. That funding ongoing is a matter for government in terms of the
new National Landcare Program, which was announced in the budget. But on top of the maintenance of those
existing IPAs, the government has announced $15 million for new IPAs, which I envisage will be in identifying
new areas and working to get an IPA on those areas.

Senator CHISHOLM: Sure. So for the next five years from 2018-19, have you got a breakdown of what
would be existing consultation versus new IPAs?

Ms Campbell: Those decisions are a matter for government and they are being worked through in the context
of the whole of the Landcare program. But the final decisions on that have not been taken. 

Ms Jonasson: I will also add—I am sorry to interrupt—that they are also being worked through with the
Department of the Prime Minister and Cabinet. We have the $15 million, which is for new IPAs. That has been
announced. But the rest of the funding really will come down to decisions there.

Senator CHISHOLM: So there is overall money for Landcare for the department. Funding out of that would
go to each of those programs?

Ms Campbell: That is my expectation, and that is the current situation, yes.

Senator CHISHOLM: And that would be a decision for government?

Ms Campbell: That is correct.

Senator CHISHOLM: So in terms of the considerations for the amount of annual funding going forward for
the existing IPA network, there is no real way to identify whether that figure is going to be reduced on a per
annum basis at the moment?

Ms Campbell: The government has not announced the ongoing figure for Indigenous protected areas, so that
is the case at the moment. We cannot compare the annual figures from now to the future, but we would be able to
do that, I imagine, once government has made the announcements on funding for that program.

Senator CHISHOLM: Minister, do you think that that money towards annual funding for Indigenous
protected areas will increase or decrease over the forwards?

Senator Birmingham: It is not for me to think about matters that will be determined either in the budget
context or portfolio context for Mr Frydenberg, but I am happy to take it on notice.

Senator CHISHOLM: Would the minister comment on whether there is still support for the national reserve
system as part of this?

Ms Campbell: I can answer that.

Senator Birmingham: The short answer is yes.

Ms Campbell: Yes. The Indigenous protected areas network makes up a very important part of the national
reserve system. They cover approximately 44 per cent of our reserves across the country. The government has
flagged the expansion and continuation of those Indigenous protected areas.

Senator CHISHOLM: And they are still regarded as part of the national reserve?

Ms Campbell: That is correct. And a very important part of the national reserve system.

Senator CHISHOLM: And what specific steps is the department taking to ensure that those Indigenous
protected areas are supported as part of the national reserve program?

Ms Campbell: So in terms of how we manage those contracts that we have for the existing Indigenous
protected areas, they are managed by the Department of the Prime Minister and Cabinet on our behalf. The
contracts that are set up with the project proponents are set out to have environmental objectives. They are funded
out of the Landcare program. We work very closely with the Department of the Prime Minister and Cabinet on
the delivery and management of those contracts and with the decision-makers on the release of the funding. So we
approve all funding milestones under those contracts. When we are talking about the national reserve system, we
work with Prime Minister and Cabinet and a range of Indigenous stakeholders on the value of that and the
continued linkage for the Indigenous protected areas to the national reserve system.

Senator CHISHOLM: Is it correct that there are currently 75 Indigenous protected areas operating across
Australia at the moment?

Ms Campbell: I will have to get the numbers. That number sounds about right. I think there are 75. We are
currently providing funding for the support and declaration of management of 70 of those properties.

Senator CHISHOLM: And 18 in planning or development stages as part of the program that is funded
through to June 2018?

Ms Campbell: I do not have the breakdown of those figures, so I would have to take on notice the breakdown
of which are in declaration and which are still in the planning phases.

Senator CHISHOLM: I am interested in the funding for those that are in planning as well and the status of
them.

Ms Campbell: So of the contracts that are in the planning stages, there is money provided from the
Commonwealth through the National Landcare Program for the consultation phase of the IPA. In accordance with
the program guidelines that the contracts were established under, when an IPA reaches its declaration stage and is
formally declared, there is an expectation that there will be increased funding, and that is a decision of the 
minister. But there have been a number of IPAs in the recent few years that have proceeded to declaration. Under
the guidelines, there is an increase in funding to allow for the increased management responsibility of that area.

Senator CHISHOLM: Minister, is that something you are confident the government will fund as the
potentially 18 new ones are agreed to—ongoing funding for those as part of the system?

Senator Birmingham: Well, I am sure appropriate consideration will be given to that. I am happy to take that
on notice in terms of any additional information that Mr Frydenberg can provide.

Ms Jonasson: I will also add that, as we recall, $15 million was announced in November last year out of the
$100 million that is specifically for new Indigenous protected areas. So that will be supporting some of this, I am
sure.

Senator CHISHOLM: Is there any timeline on when you think the government will make a decision about
the future funding of the IPA program for June 2018 to June 2023?

Ms Campbell: This was part of the $1.1 billion announced in the budget. As I flagged on another issue earlier
in the committee, we are working very closely with the government to get the details finalised. We expect
announcements shortly. We are very conscious that contracts expire in 14 months or so. We need to get those well
underway early. So we are hoping for an early announcement.

Senator CHISHOLM: Would the minister have anything to add to that in terms of the timing of the decision?

Senator Birmingham: I think decisions will be made under the process that has been outlined. As, of course,
Ms Jonasson has indicated, there is $15 million additional that was indicated and is under part of a formal
consultation process.

Senator CHISHOLM: And the additional money was through the deal around the backpacker tax, from
memory, at the end of last year?

Ms Campbell: That is correct, yes.

Senator CHISHOLM: And is that going to be an annual funding stream for those five years?

Ms Campbell: The funding for the $100 million, which was provided out of MYEFO, was funding over four
years from 2016-17. That is the funding that the government has announced. Further detail on the funding for the
new Indigenous protected areas is being worked through. We are working very closely with Prime Minister and
Cabinet on developing that to provide advice to government.

CHAIR: We are just about out of time for this session. Have you got many more questions?

Senator CHISHOLM: No, not that many. In terms of the process for the establishment of new IPAs, are they
considered against set criteria?

Ms Campbell: We will work with Prime Minister and Cabinet and determine the process for setting up the
new Indigenous protected areas and advise government on that. There will be, I expect, some kind of process and
some kind of criteria, but we have not got to the stage of considering those and advising government.

Senator CHISHOLM: Can the department provide a funding figure for the Indigenous Rangers program over
the five years from 2018-19?

Ms Campbell: The Indigenous Rangers program is administered and delivered out of the Department of the
Prime Minister and Cabinet, so you would have to ask that department for those figures. We do not have those
figures.

Senator CHISHOLM: So in terms of the department's role with the Indigenous Rangers program, how does
that gel together?

Ms Campbell: The Indigenous Rangers program, as I flagged, is run out of the Department of the Prime
Minister and Cabinet and is a matter for that portfolio. We work very closely with Prime Minister and Cabinet on
a range of linkages, including through the Indigenous protected areas. Minister Scullion announced last week
additional funding for compliance training for rangers, for example. This portfolio is working very closely with
the Department of the Prime Minister and Cabinet on that funding—for example, how those compliance powers
for the rangers can apply both under the EPBC Act and through our national parks.

Senator CHISHOLM: Is there any aspect of it that is currently funded from the National Landcare Program
biodiversity fund?

Ms Campbell: There is no funding provided under the biodiversity fund for this portfolio. My recollection is
that any contracts under the biodiversity fund were transferred to the Department of the Prime Minister and
Cabinet in 2014 as part of the machinery of government changes. We do have some payments made for the ranger
program under the Landcare program. Those arrangements continue. We pay a relatively small amount of 
supplementation money for the ranger program, which is in many ways a legacy of when we managed that
program within this portfolio.

Senator CHISHOLM: Is that acting as sort of coordinator funding for the program? Is that correct?
Ms Campbell: The funding that we provide for rangers is really effectively paying some of the milestones for
the contracts that are managed by the Department of the Prime Minister and Cabinet.

Senator CHISHOLM: So how much is that?

Ms Campbell: Over the four years of the National Landcare Program from 2014 to 2017-18, it is $34.729
million.

Senator CHISHOLM: To 2017-18?

Ms Campbell: That is correct.

Senator CHISHOLM: And what are the plans to sustain that post the end of next financial year?

Ms Campbell: Again, the government's decisions on the National Landcare Program and the detail to be
funded out of the next suite will be a matter for government. We expect announcements shortly. The Department
of the Prime Minister and Cabinet funds the majority of Indigenous rangers through the Indigenous advancement
strategy and has extended contracts for the rangers under that program. Last week, it announced further money for
compliance training. So questions about the Indigenous Rangers program really should be directed to that
portfolio.

CHAIR: I have one final question for Mr Andrews. It relates to something that we have talked about before,
and that is the terrible situation with feral dogs in Western Australia and the havoc that they are playing on many
farms and pastoral stations. Can you give us an update on the program there? I have had feedback that the 1080
baits are not working. In fact, the goannas are eating them and the dogs are not. Can you perhaps update us, or
take it on notice, because I would like quite a comprehensive overview of what is happening with the wild dog
program in Western Australia.

Mr Andrews: Actually, I will take it on notice and consult my colleagues in the agriculture department
because they provide a large amount of funding to tackle feral dogs. Some of the baits that we have been
distributing—the Eradicat baits—are also controlling feral dogs to some extent. With the goannas, one thing we
do not have to worry about is the goannas actually suffering because, particularly in Western Australia, they are
immune to 1080. It is a biodegradable substance found in the leaves of some of the gastrolobium plants there. But
obviously that does not mean it is not getting a result for the dogs. So I will take it on notice for you.

CHAIR: Thank you very much. Any more questions for this program? If not, thank you very much. The
committee will now suspend for afternoon tea until 4.20 pm. We will resume with program 1.2.

Proceedings suspended from 4.02 to 4.23 pm

CHAIR: This hearing will now resume. I call officers from the department in relation to program 1.2.

Senator URQUHART: In terms of environmental accounting, can you outline the progress towards meeting
this objective?

Mr Thompson: In November 2016, obviously the Commonwealth, state and territory environment ministers
agreed to pursue a common national approach to environmental economic accounting. For us, that is a framework
for capturing and organising information on the environment and its contribution and its interaction with
economic activity and the impact of that economic activity on the environment. We will be over the course of this
year—and we are this year—developing a strategy on a common national approach to environmental economic
accounts to provide to environment ministers by December 2017. That is the time line they set. As part of that, we
are collaborating with other governments and agencies within the Commonwealth.

Senator URQUHART: So you mean state governments?

Mr Thompson: And state governments; yes, that is right. State and territory governments and other agencies
within the Commonwealth family of departments.

Senator URQUHART: How many agencies, Mr Thompson, do you have to link over?

Mr Thompson: Within the Commonwealth?

Senator URQUHART: Yes.

Mr Thompson: Or within the states and territories?

Senator URQUHART: Pretty much within the Commonwealth. 

Mr Thompson: So those data-rich agencies and those who have expertise in environmental economic
accounting. So primarily the Australian Bureau of Statistics are close partners. Geoscience Australia are
significant data holders and data analysts. The Bureau of Meteorology in our own portfolio has responsibilities in
particular to meteorological reporting but also water accounts and environmental information. And there is
CSIRO. We will also be engaging with the Department of Agriculture and Water Resources and, to the extent
relevant, the Department of Industry, Innovation and Science. We are also—

Senator URQUHART: There cannot be too many left off that list.

Mr Thompson: No. It is comprehensive. But they are probably the major entities at Commonwealth level
who have an interest in natural resources and natural resource management, yes.

Senator URQUHART: I want an outline of progress towards meeting the objective of greater consistency
and coordination in the gathering and presentation of environmental data and information. How is that going?
How are you going towards meeting that objective?

Mr Thompson: I think there are three things that we are doing towards that end. Colleagues might add a bit in
terms of specifics on environmental data itself. One is, as I said, developing a strategy, which we will want to
provide for nationally consistent reporting on Australia's environment.

Senator URQUHART: Mr Thompson, is that time line the same as the accounting—December?

Mr Thompson: It is, yes. So I am kind of answering your question in the light of environmental economic
accounts. Is that what you are asking?

Senator URQUHART: Yes. Accounting and data and information.

Mr Thompson: I will start with the accounting lens and then we will move on to broader data things.
Senator URQUHART: Great. Thank you.

Mr Thompson: As I mentioned, we will also, through the development of that strategy, be looking to run
some pilots in terms of accounting. We are in discussions with the ABS and with state and territory governments
around that and with other important stakeholders who have been progressing accounts work outside of
government, including the Wentworth Group and their accounting for nature approach; the Australian National
University; and CSIRO. We will also be organising, as the environment ministers asked us to, a national
workshop on environmental economic accounting that will bring together practitioners and seek to develop that
approach.

Senator URQUHART: And when would that workshop be held?

Mr Thompson: At this stage, in the third or fourth quarter of this year—probably the third quarter of this
calendar year.

Senator URQUHART: And are you expecting recommendations or any information to come out of those
workshops to finalise or get into that strategy?

Mr Thompson: Yes.

Senator URQUHART: Is that part of the purpose?

Mr Thompson: That is exactly right. We are looking to the workshop to help inform and populate the strategy
before it goes to ministers.

Senator URQUHART: Will the workshop be by invitation?

Mr Thompson: It will be by invitation through very much an expert workshop. But it will be to
governments—state governments, our own government agencies—and to external practitioners, including the list
that I gave before of the Wentworth Group and natural resource management bodies that have been doing some
accounting on the ground. In terms of data more broadly, there are a couple of other initiatives that we have been
running. The data integration project, which is being run out of the Department of the Prime Minister and
Cabinet, has identified an environmental analytics hub, and we are in discussions with the Department of the
Prime Minister and Cabinet and other departments about hosting that hub in our department. That is an important
development in terms of data integration. We can talk a little more about that. The essential environmental
indicators are another stream of work that we have running from the department—as I think I said earlier in the
hearings today—to identify those key indicators and get some coalescence from governments and other interests
in natural resources around those indicators which we need to track as a corps over time.

Senator URQUHART: And do you have specific objectives that you will be tracking? What are they?

Mr Thompson: In terms of the essential environmental work? 

Senator URQUHART: Yes.

Mr Thompson: We have started by looking at some of the terrestrial environment ecological assets because
we see that as a priority. Dr Terrill might want to add something in that respect.

Dr Terrill: In terms of the essential environmental measures program, there has been a lot of work with all the
stakeholders that Mr Thompson has mentioned. It is still converging on exactly what measures are the best to
identify changes. In a broad sense, you can put them into stock and flow categories, which I am sure are very
familiar to you, but there is not yet a settled group of those. Obviously, that work is a precursor that will feed
strongly into the environmental economic accounts that Mr Thompson has referred to.

Senator URQUHART: Great. Thank you.

Senator WHISH-WILSON: I want to ask some questions about Japanese whaling in the Southern Ocean.

Mr Thompson: That used to be in 1.2. It is now in program 1.4, I think.

Senator Birmingham: Mr Thompson, if you cannot say it with certainty, who possibly can.?

Senator WHISH-WILSON: I was advised 1.2, but I would be happy to ask it in 1.4.

Dr de Brouwer: Senator, it is on the new outcomes. Whaling is in 1.4.

Senator Birmingham: Well done.

Senator WHISH-WILSON: That means we can probably move to 1.4 very soon.

CHAIR: In fact, if there are no more questions, we could move to 1.4 now. If there are no other questions for
program 1.2, I thank the officials. We will now move to program 1.4, which is conservation of Australia's heritage
and environment.

[16:32]

Senator WHISH-WILSON: I will kick off with a few questions on whaling. My understanding is that Japan
will have to submit its proposed catch to a scientific working group within the International Whaling
Commission. Are we part of that working group going forward?

Mr Oxley: I need to say at the outset that we are a bit disabled in terms of our ability to assist the Senate
today, so bear with us. The disablement is that Nick Gales, Australia's whaling commissioner, is on his way. He
was expecting to be here later in the evening, and the program has come forward. So I apologise for that.

Senator WHISH-WILSON: Do you want me to go to another line of questioning and come back to whaling
when he arrives?

Mr Oxley: I will have to see whether we can get a communication to him that will even have him here in time
today. But if you want to go to another line of questioning while we try to achieve that, I would appreciate it
sincerely.

Senator WHISH-WILSON: Sure. I might ask some questions about the EPBC status of great white sharks.
The committee does have an ongoing Senate inquiry into this issue and the broader issue of shark mitigation. I
want to ask some questions about specific comments in the media recently. Just confirm that the recovery plan for
the great white that was initiated on 6 August 2013, the EPBC recovery plan, is due for review in 2018.
Mr Knudson: The act itself?

Senator WHISH-WILSON: The recovery plan, yes.

Mr Richardson: Yes, that is correct. Its five-yearly review will be due in 2018.

Senator WHISH-WILSON: When do you start that process? Has it already started?

Mr Richardson: It has not started as yet.

Senator WHISH-WILSON: Any idea when you will be able to report the effectiveness of the recovery plan?

Mr Richardson: Well, by the deadline in 2018. I am not trying to be cute, but we have not actually scheduled
it as yet. But it will be done. Presumably later this year we will commence that.

Senator WHISH-WILSON: Is the work being done by CSIRO on population levels of great white sharks
going to be a key part of informing that recovery plan?

Mr Richardson: All information that is available at the time of the review will be taken into account. So
assuming—

Senator WHISH-WILSON: I am asking whether it is a key part of that information.

Mr Richardson: It will be a very important part, yes. Information from CSIRO is highly valued, of course. 

Senator WHISH-WILSON: That is good enough. Has your department provided the minister with any
advice on population levels of great white sharks in recent times?

Mr Richardson: We have provided some information that CSIRO released, I think, in 2014 on the population
estimates at that time of the east coast population, which is the only available information at this point.
Senator WHISH-WILSON: Is it your understanding that the work being done by CSIRO will update that
2014 report with new population data?

Mr Richardson: That would be a question to ask of 1.2. But my understanding—that research project is being
done under our National Environmental Science Program—is that the white shark project, which has CSIRO and
other participants, is looking to refine that east coast population estimate from 2014. So it will refine that for the
east coast. My understanding is it is going to deliver a point estimate of the population size of the west coast
population, but I suspect with bounds around it. It will be a range estimate, if you like.

Mr Knudson: I just want to add that it will also cover off the east coast population.

Senator WHISH-WILSON: That is correct. That is my understanding. It will potentially break down
juveniles with older sharks, more mature sharks, as well?

Mr Richardson: So my understanding—I am not sure if my research colleagues are still here—is that the
estimates for both the east and west coasts will be for the adults and not a total population estimate. It will be an
estimation of the adult population size.

Senator WHISH-WILSON: Okay. I do not want to pre-empt that report.

Mr Richardson: Sorry. If I need to correct that, I will correct it later.

Senator WHISH-WILSON: That is fine. Did the 2014 study that you provided to the minister show that
population levels of great white sharks have been increasing? I underline the word 'increasing'.

Mr Richardson: My understanding is that we have not had a population estimate before, so that is a
population estimate at a point in time.

Senator WHISH-WILSON: That is correct. That is what I was hoping you would say.

Mr Richardson: That is the 2014 research advice.

Senator WHISH-WILSON: So unless we get something versus 2014, we do not know whether the
population levels are increasing per se. We know there is a recovery plan in place because, going back over
decades, there were concerns about numbers of great white sharks. That is part of an international treaty on
protection. Mr Frydenberg, our environment minister, said not long ago, on 20 April 2017 that 'Blind Freddy'
could see that there were more 'great white sharks in the water and people's lives would be put in danger if we
don't take action'. Where would the minister have got that idea that blind Freddy could see that white shark
populations were going through the roof?

CHAIR: I think you just have to talk to any cray fishermen, fishermen or anyone else who is off the coast of
Western Australia and they will tell you the same. That is why this survey is so important.

Senator WHISH-WILSON: Thank you, Senator Reynolds. I would prefer to hear from the department
whether they believe it is—

CHAIR: My apologies, Senator Whish-Wilson.

Senator WHISH-WILSON: That is okay. Actually, Senator Reynolds makes a good point. Perhaps, Senator
Birmingham, I could ask you. Is it anecdotal evidence from cray fishermen off the coast of Western Australia that
the minister is using to make that comment?

CHAIR: They are in the water all day every day.

Senator WHISH-WILSON: It sounds very scientific to me. I was just interested if that might be what is
informing the federal environment minister.

Senator Birmingham: Senator Whish-Wilson, I am happy to seek some information from Mr Frydenberg's
office as to the evidence base used by the minister.

Senator WHISH-WILSON: Could anyone in the department comment on why our federal environment
minister, whose role it is to protect the environment, is saying that blind Freddy could see that white shark levels
are going through the roof?

Mr Oxley: Senator, I do not think we have anything to add beyond the undertaking the minister has given to
seek advice from Minister Frydenberg.

Senator WHISH-WILSON: Can the minister unilaterally change an EPBC listing—let us talk about white
sharks as an example—without it first going to the threatened species committee? Could somebody talk about the
process there?

Mr Richardson: The process to add a species to the list, remove a species from the list or change a species's
status on the list—from endangered to critically endangered, for example—all requires the minister to seek
advice, add it to an assessment list and then have the Threatened Species Scientific Committee provide advice on
the status of that species.

Senator WHISH-WILSON: Have there been any requests from the minister or otherwise to review the status
of the great white shark? Have there been any discussions about a referral to that committee on its listing?
Mr Richardson: I am not aware. Certainly it has not been nominated and has not been added to the list. There
has been no briefing to that extent.

Mr Oxley: In terms of the steps in the process, each year there is a call for public nominations made. That is
made towards the end of the calendar year. The Threatened Species Scientific Committee then goes through a
process of assessing and looking at all of those nominations. It then puts to the minister what is called a proposed
priority assessment list. The minister considers that list, and the minister may adopt the advice of the Threatened
Species Scientific Committee, which may vary the species that he wishes to have added to what then becomes the
finalised priority assessment list. We are in the midst of that process at the moment. So what is called the FPAL
for 2017 is yet to be determined.

Senator WHISH-WILSON: So you would put it to him rather than the other way around? He would not
come to you?

Mr Oxley: The minister has the discretion to add species to the finalised priority assessment list should he
wish to do so.

Senator WHISH-WILSON: Interesting.

Mr Richardson: Just to correct that, it is not the department that provides that advice to the minister on a
proposed list. It is the committee.

Senator WHISH-WILSON: Okay. Firstly, would it be likely that that would occur prior to the recovery plan
being finalised for the great white shark? With you recommending that to the committee or at the discretion of the
minister to the committee, would you expect that the recovery plan would be completed before any kind of—

Mr Richardson: There is a current recovery plan, a relatively recent one from 2013.

Senator WHISH-WILSON: Which is due in 2018.

Mr Richardson: So it is due to be reviewed. As to whether it needs to be amended or not, there will be advice
come from the committee to the minister about whether the plan still has the right actions and measures et cetera
or whether it needs to be revised or replaced.

Senator WHISH-WILSON: That is right.

Mr Richardson: So the current plan does not expire for 10 years.

Senator WHISH-WILSON: Specifically my question is in relation to the recovery plan 2018 and the fiveyear
review that we talked about earlier. Would you expect that you would wait for that review to occur before
this was recommended to the committee or vice versa or the minister used discretion?

Mr Oxley: The recovery planning provisions operate separately from the listing and assessment provisions.
The recovery planning provisions do not in any way fetter the minister's ability to adjust over time the finalised
priority assessment list.

Senator Birmingham: Senator Whish-Wilson, just in terms of the great white and its place on the list, I draw
your attention to a column published by the minister in the Weekend West over the weekend—

Senator WHISH-WILSON: Which I have here.

Senator Birmingham: where he steps very clearly through the obligation to obtain advice from the
Threatened Species Scientific Committee, the population survey work that CSIRO would have to undertake as
well as the interaction with the convention on the conservation of migratory species and, of course, the rules
under that convention. Australia has no intention of rescinding or changing its adherence to those rules.
Senator WHISH-WILSON: Thank you, Senator Birmingham. I have a copy of it here. I have read it. I have
some questions about that very shortly. 

Senator Birmingham: A number of questions you seem to be asking are already addressed by the minister in
that. I am pleased to see that you have a copy of it and that you have read it.

Senator WHISH-WILSON: Thank you for assisting me with that, Senator Birmingham. However, Mr Oxley
just said that the minister has discretion to prioritise the delisting of a species. Did I get that correct?

Mr Oxley: No. I said the minister has discretion to ask the Threatened Species Scientific Committee to
undertake an assessment.

Senator WHISH-WILSON: Sorry. That is what I meant. I apologise. He has the discretion to ask the
committee to prioritise an assessment?

Mr Oxley: That is correct.

Senator Birmingham: That is what he says in the article.

Senator WHISH-WILSON: Without the recovery plan having been completed. That is true?
Mr Oxley: Yes. That would be true.

Senator WHISH-WILSON: If the recovery plan or the CSIRO report is going to be an important part of
providing information around the recovery plan, is it possible that if the recovery plan shows a recovery in
population levels, that that could affect the decision around the delisting of the species?

Mr Oxley: I would not characterise it in terms of the recovery plan per se. The Threatened Species Scientific
Committee, if it were reviewing the listing status of any particular species—in this case, white sharks—would be
looking at what population information there is to determine whether there had indeed been a recovery in the
population of that species. That would be a quite, I would expect, detailed piece of analysis done on the best
available information. TSSC would form its advice to the minister and give it.

Senator WHISH-WILSON: So it is feasible, then, that if the populations were seen to have recovered, which
I would have thought is a good thing for healthy oceans, that may trigger a potential delisting of the species as a
threatened or protected species?

Mr Oxley: As a general proposition, a recovery in a species, we would always hope, would lead to the
downlisting of the level of protection of that species and, ultimately, in the face of success, the removal of that
species from the listing under the EPBC Act.

Senator WHISH-WILSON: So what we would be looking for, then, would be clear evidence that population
levels had increased over time and you being comfortable they were at a level that means they were not
threatened any longer. Would that be a reasonable position?

Mr Oxley: That would be a core component of any advice or assessment that was being undertaken.
Senator WHISH-WILSON: Thank you very much for that. I have a few more quick questions on this.

CHAIR: To clarify, I understand from the minister's column that CSIRO is currently doing a study into shark
populations on the east and the west coasts. Are you able to provide a little more information on that study and
how that might feed into this?

Senator WHISH-WILSON: I have asked this question.

Mr Oxley: Mr Richardson indicated earlier that that is something that is being run out of the National
Environmental Science Program in outcome 1.2. We can certainly take that on notice.

CHAIR: Thank you.

Senator WHISH-WILSON: I understand that Australia is a signatory to the Bonn convention on migratory
species. What is the impact of being a signatory on Australia's obligations regarding great white conservation?
Mr Richardson: You are correct. We are a signatory to the Convention on the Conservation of Migratory
Species of Wild Animals. Your question was what are the implications of being a signatory?

Senator WHISH-WILSON: Yes. What is the impact of being a signatory regarding our obligations around
white shark conservation?

Mr Richardson: The convention puts obligations on signatory countries to cooperate with other range states
for listed migratory species that we are a range state for. That is for appendix 1 and appendix 2 listed species. For
appendix 1 listed species, it also provides for a prohibition on take with certain exceptions.

Senator WHISH-WILSON: Could you speak very briefly on what the prohibition on take means exactly?

Mr Oxley: Before Mr Richardson responds to that specifically, the way our obligations under the convention
on migratory species are given legal effect is through the Environment Protection and Biodiversity Conservation 
Act. So all species which are listed under the convention on migratory species are protected as matters of national
environmental significance under the EPBC Act.

Senator WHISH-WILSON: Okay. So in relation to the prohibition on take, what kind of exceptions does that
allow for?

Mr Richardson: There are various exceptions. The one that springs to mind is the one around exceptional
circumstances, and that is used by quite a significant number of the signatory countries. Australia has used it in
the past. But the fundamental nature of being a signatory to the migratory species convention is that we are
seeking to cooperate with other countries on that. We have a large number of migratory species that are listed and,
as Mr Oxley said, are protected as matters of NES under the EPBC Act as migratory species. We cooperate with
range states globally. For white sharks, it is certainly in the southern hemisphere; for migratory shore birds, it is
up through the flyway into Asia; for turtles, it is with South America and other countries.

Senator WHISH-WILSON: I understand that, Mr Richardson. I am interested in whether exceptional
circumstances would essentially, for want of a better term, be a loophole for killing white sharks. Could you
justify that?

Mr Richardson: I do not think it would be characterised as a loophole. Certainly for white sharks globally,
there are a number of signatory countries where there is a level of take to protect beachgoers, including in
Australia.

Senator WHISH-WILSON: Can you give us more information, perhaps on notice, as to examples overseas
and whether there is any quantification of what that might be.

Mr Richardson: Sure.

Senator WHISH-WILSON: I am quite interested in what our obligations would be. My last couple of
questions on this do relate to the article in the West Australian that Senator Birmingham raised. In fact, as Senator
Birmingham said, the minister said that both under EPBC law and referral to a committee prioritisation of a
referral to the committee would be a step. He talks about our obligations. On the other hand, the minister then said
that the Western Australian government should consider what is happening in Queensland and the New South
Wales coastline, which is nets and drumlines. Does that suggest that if Western Australia applied to put in shark
nets and a drumline program, they would do that under an EPBC approval process, or is the minister referring to
using section 76 there?

Mr Oxley: I do not think it would be right for the department to project any particular proposed way that that
would be addressed into the minister's comments except to say that the minister has indicated a willingness should
the Western Australian government put forward a proposal to consider it under the relevant provisions of the
EPBC Act.

Senator WHISH-WILSON: Let me reword that, Mr Oxley. Previously under the Barnett government, the
EPA in Western Australia conducted a study into an ongoing program using drumlines to cull sharks. Their
advice was that it would not pass an EPBC approval process. On that basis of the EPA report, would you be
demanding from the Western Australian government an EPBC approval process?

Mr Oxley: There is not a proposal in front of the department to consider and provide advice to the minister.
The minister has invited the Western Australian government to make such a proposal, so I do not know that we
can really step further into that area.

Senator WHISH-WILSON: Could you clarify that it would have to be an exemption under section 76, such
as we saw in—

Senator Birmingham: Senator Whish-Wilson, until a proposal is put forward, in a sense, at that stage, the
determination, as I understand it, is made as to whether a proposal is a controlled action. Obviously, yes, we could
go through hypotheticals of whether such an action would be a controlled action or otherwise. I guess you can
draw your own conclusion from some of the precedents that have existed recently. The minister, in his opinion
piece, made it clear that, I guess, and implied that he considered that it may be because he indicated that he would
have to and would give consideration to any such request. But as for whether or not it would be a controlled
action, you would have to see the actual nature of the act, and then that process would be triggered.

Mr Oxley: I will make one last comment in addition to the minister's. I make the point that these questions
about the actual operation of the EPBC Act assessment and approval provisions are most appropriately dealt with
under outcome 1.5, not under 1.4.

Senator WHISH-WILSON: We have certain questions under 1.5. But this is about the EPBC recovery plan
and potential delisting. 

Mr Oxley: Yes. Which we have addressed, yes.

Senator WHISH-WILSON: Well, yes, you have. Perhaps not to my satisfaction. But, nevertheless, that is to
be expected at estimates. I might stop there. I have other lines of questioning later.

Senator LINES: You may have answered this in response to Senator Whish-Wilson, so my apologies if you
have. Can you tell us how much has been invested in researching the western white shark populations?

Mr Richardson: I am afraid we will have to take that on notice. I am not sure what the split is between east
and west. As I have mentioned earlier, this is under outcome 1.2, the National Environmental Science
Programme, so we will have to take that on notice.

Senator LINES: I knew that you skirted around. I was not quite sure. We recently had an inquiry in Western
Australia that looked at this issue of sharks. The state minister, Minister Kelly, appeared before the committee. He
indicated that he had written to Minister Frydenberg seeking research dollars and commitment. Has the
department seen that letter?

Mr Knudson: I will go back to your earlier question. I do have the figure here for the study that is being done
under the National Environmental Science Programme with respect to shark populations. This is both eastern and
western populations. The Commonwealth has invested $764,000 into that project. That is being done in
conjunction with the Western Australian government.

Senator LINES: Right. But that $764,000 is split between the east and west?
Mr Knudson: That is my understanding.

Senator LINES: So who wants to answer my second question?

Mr Oxley: The department is aware of that letter. We have seen it. I am not quite sure where we are at in
terms of advising the minister on a response to that request.

Senator LINES: So, as far as you know, the minister has not yet responded to the Western Australian letter?

Mr Oxley: This is to my knowledge. We will need to clarify and come back.

Senator LINES: Thanks.

Senator WHISH-WILSON: I want to ask for a point of clarification on Senator Lines's questions. Can you
give us, perhaps on notice, what the quantum of funding has been overall for research specifically around shark
mitigation and the tagging programs?

Mr Oxley: We certainly can attempt to.

Senator WHISH-WILSON: If you could. I am trying to get an idea of how much we have invested in the
public good.

Mr Richardson: I want to clarify the question. Are we talking about oceanic sharks?

Senator WHISH-WILSON: Specifically the white sharks. I know you tag lots of other sharks, but white
sharks and bull sharks.

Mr Richardson: Okay.

Senator WILLIAMS: What work is CSIRO currently doing in terms of assessing numbers on both the west
coast and east coast as far as shark numbers go?

Mr Richardson: White sharks?

Senator WILLIAMS: Yes.

Mr Richardson: I think we have covered some of this. My understanding is that there is the National
Environmental Science Programme. Mr Knudson just gave a quantum for some of that, at least. That research on
the east coast has already produced in 2014 a point estimate of the size of the adult population on the east coast.
The end of that research project is the end of this calendar year. I understand that they are going to refine that
population estimate. On the west coast, they are looking to produce their first adult population estimate by the end
of this calendar year.

Senator WILLIAMS: The end of the year; right. Has the New South Wales government used smart
drumlines and nets?

Mr Richardson: They do, yes.

Senator WILLIAMS: What has the effectiveness of this program been as far as the New South Wales
government's activities that have been carried out? 

Mr Knudson: We could go with that at 1.5, because that division has had regular interaction with the New
South Wales government and, quite frankly, the WA government.

Senator WILLIAMS: We will leave it to 1.5.

Mr Knudson: That is correct.

Senator WILLIAMS: I will tag it on then, thanks, Chair, at 1.5.

Senator URQUHART: I have some questions on World Heritage listing. At the last estimates, Budj Bim was
raised; I think that is how you pronounce it. Can you please give me an update on that World Heritage listing and
where it is at?

Mr Johnston: Budj Bim has been added to Australia's World Heritage tentative list. At the moment, the
Victorian government is putting together a nomination dossier. They are also undertaking some field research to
support that dossier. We will work with them in the second half of this year with the aim of finalising that to
submit the formal nomination.

Senator URQUHART: And when will that be, sorry?

Mr Johnston: We are hoping and aiming to have it done before 1 February 2018. Whether we get there or not
is yet to be seen, but that is the intent, or the hope, at this point.

Mr Oxley: I might add that if we make that deadline—and we certainly expect to be able to do that—that
would enable the World Heritage Committee to be considering that nomination formally at its meeting in the
middle of 2019. So, once submitted, there is a process that runs over the course of about 16 months or thereabouts
for that nomination to be formally assessed and then the proposal finally to be put to the World Heritage
Committee.

Senator URQUHART: Are there any other submissions that will be on the tentative list?

Mr Johnston: There are two other places on the tentative list, both of which date back some years. They are
extensions to the Gondwana Rainforests World Heritage site, which straddles New South Wales and Queensland,
and then an extension to the Fraser Island World Heritage listing. The 2015 meeting of environment ministers
discussed the tentative list update and identified that process in their communique. There was mention as well of
Cape York in Queensland, Royal National Park in New South Wales and the West MacDonnells in the Northern
Territory. But the process is that each of those state governments will need to work up a credible case for World
Heritage listing before the Commonwealth would brief the minister about adding it or not to the World Heritage
list.

Senator URQUHART: Can you give me an update on national heritage?

Mr Johnston: Yes. Any particular element?

Senator URQUHART: Just a broad update.

Mr Johnston: Since we last appeared, we have had two new listings to the National Heritage List. These are
the Cornish mining sites in South Australia, Burra and Moonta. They were added by the minister this month. In
terms of the work plan, we have 15 places being assessed for the National Heritage List and we have another
three where the Heritage Council has made its final recommendation. We are just putting together the paperwork
before being able to provide briefing on them to the minister. The other element that is probably pertinent is
similar to the finalised priority assessment list for threatened species. We have a similar process for heritage. The
Australian Heritage Council has been considering national heritage nominations. We expect the minister will set
the finalised priority assessment list for the National Heritage List before 1 July.

Senator URQUHART: This year?

Mr Johnston: This year.

Senator URQUHART: Can you give me an update on any research into cultural heritage and built heritage?
Mr Johnston: We have not been funding any specific research ourselves.

Senator URQUHART: So there is no funding allocated to those processes?

Mr Oxley: We are not a research funding department in relation to heritage. The primary research that we do
is around the assessment of nominations and places for inclusion on the National Heritage List. So that is where
we do do some research. The other area where a small investment is made is, again, through the Australian
Heritage Council, where some thematic studies have been done in recent years looking at, for example, deserts as
a theme for potential identification of places for inclusion on the National Heritage List or into that process. That
would be, I think, the true extent of any research that we do. 

Mr Johnston: The thematic study they did before deserts was on rock art places.

Senator URQUHART: Yes. Are there any staff? If there are, how many are fully allocated to World Heritage
in the department other than the Great Barrier Reef work and to heritage?

Mr Johnston: It is difficult to give definitive answers because most staff tend to work on a combination of
World Heritage, national heritage and sometimes other acts as well that we administer. Generally speaking, as of
1 May, and within the wildlife heritage and marine division, we had approximately 43 staff who worked on some
heritage related matters.

Senator URQUHART: Forty-three?

Mr Johnston: Forty-three. That is not necessarily the entire job but some part of their job.

Senator URQUHART: That is not fully allocated?

Mr Johnston: No. That is right.

Senator URQUHART: They have some responsibility?

Mr Johnston: That is correct. Of those, some 26 had some involvement with World Heritage matters but,
again, not a full-time role.

Mr Oxley: Mr Johnston's branch is running at about 36 ASL. But there are part-timers and what have you in
that. That is why it is really hard to put concrete numbers around it. But that is the scale at which we engage in the
heritage. There is a team in marine and international heritage branch that has primary responsibility for our
relationship with the World Heritage Committee and engagement with the World Heritage property managers. So
that is where the number of a little over 40 comes from, all told.

Senator URQUHART: Okay. Mr Johnston, I want to get some clarification on the tentative list. Is the
government or the department not seeking to develop a tentative list?

Mr Johnston: The process is effectively an iterative one and is done in conjunction with the states and
territories through the meeting of environment ministers. Certainly states can request to have further discussions
on the tentative list.

Senator URQUHART: So who develops the tentative list to begin with? Is that a joint proposal developed by
a state or is it joint Commonwealth-state? How does it work?

Mr Johnston: Essentially, it is the Commonwealth as a state party to the World Heritage convention and it is
the Commonwealth environment minister who nominates places for the tentative list. But, in practice, the way
that we are managing it is collaboratively with the states and territories. This reflects something that Mr Oxley
might want to expand upon, which is the capacity of the World Heritage centre to consider nominations. We want
to be sure that in putting forward one nomination we are not crowding out another potential nomination without
everybody having a chance to consider them all as a whole.

Senator URQUHART: So that consideration as a whole is done with the Commonwealth and the state
ministers?

Mr Johnston: That is right. And there was a discussion in—

Senator URQUHART: Is that like a COAG process? How does that work?

Mr Johnston: It is not COAG, but it is the environment ministers forum—

Senator URQUHART: Yes. So it is sort of that?

Mr Johnston: Which is a similar process. They had a discussion in December 2015. We understand that one
of the states has requested another discussion this year on the tentative list. So there may be one later this year—a
second discussion on that.

Mr Oxley: Mr Johnston has referred to what happens within the World Heritage system. I think I may have
covered this in testimony at the last estimates hearing. The World Heritage system is resource poor and resource
hungry. It has been grappling for a number of years with the ever-growing size of the World Heritage list and the
enthusiasm of states parties to bring forward new places for inclusion on the list. It is trying to get the balance
right between bringing new places on to the World Heritage list and then looking into the management of those
places that are already on the list. Over a number of years, through discussions in the World Heritage Committee,
it has taken an attitude that those countries that are already well-represented on the World Heritage list should
take a back seat for a while and let countries that have very few or no places on the World Heritage list take the
opportunity to bring forward nominations. In that context, our assessment over the last few years has been that it
would be realistic for Australia to bring forward three or four nominations over the course of a decade for places
to be included on the list, whether that is new places or extensions of existing places. 

Senator URQUHART: So is an extension treated differently to a new listing?

Mr Oxley: An extension would be treated the same. It has to go through all of the same processes unless it
falls within the category of a minor boundary change. If it is a minor boundary nomination, the process would be
more truncated but still would involve deep consideration and still need to demonstrate that the extended area
does have outstanding universal value. So the tests are the same.

Senator URQUHART: Thanks. Can you give us an update on Cape York and, within that, the consultation
with the Indigenous people?

Mr Johnston: That is a process that is being run by Queensland.

Senator URQUHART: So the Queensland state?

Mr Johnston: The Queensland state; that is right.

Senator URQUHART: So there is no involvement from the Commonwealth on that?

Mr Johnston: No. We are not involved in the consultations. We were having some discussions early on with
Queensland about the scoping of their project, but I do not think we have been having those for the last couple of
months. I can take that on notice and confirm.

Senator URQUHART: That would be great. Any update on the national heritage listing for the Coral Sea?

Mr Johnston: Yes.

Senator URQUHART: When is it due to be finalised? Is there a deadline for assessments?

Mr Johnston: Its current due date is 30 June this year, but that can be extended until 30 June 2019. The
minister has the power to extend assessment deadlines for up to five years cumulative. So 2019 would be the final
extension date. With regard to how it has been going, the Australian Heritage Council discussed the assessment in
May 2014. They considered at that time that they did not have sufficient information to assess the Coral Sea for
its historic and natural values as they felt that while it was likely to have outstanding historic values for its
shipwrecks and that some areas of the Coral Sea may meet the criteria for natural heritage value, it was unlikely
the entire Coral Sea would meet the national heritage criteria. Following that discussion, they did not continue to
do any more work on it at that point. But, in recent months, the department has decided to scope out a shape of
how an assessment might look. I am checking if we have it scheduled. In fact, the upcoming Heritage Council
meeting is scheduled at this stage to consider a scoping for that assessment.

Senator URQUHART: So, with that scoping, is it possible to look at part of the Coral Sea rather than all of
it? How would that look?

Mr Johnston: What the council does in those scoping discussions is look at the nominated boundary and then
where the heritage values lie. Sometimes it might choose to look at a smaller or a larger boundary depending on
where it believes the heritage values would lie.

Senator URQUHART: You said they are meeting to do that. When are they meeting?

Mr Johnston: Their meeting is at the end of June.

Senator URQUHART: That is at 30 June?

Mr Johnston: On 22 or 23 June. I have not got a copy of their draft agenda with me, but our tracking sheet
has them at this stage considering a scoping document.

Senator URQUHART: Great. How often do heritage officials meet with communities who want to progress
heritage assessments?

Mr Johnston: It depends. We do not have any formal consultations with communities as such. So normally it
would depend on if people contact us and ask to have a meeting. That is outside the formal consultation processes
we run for places already being assessed.

Senator URQUHART: Does it happen very often, Mr Johnston, that you get communities call you and want
to discuss that?

Mr Johnston: It is not that common, but it does happen from time to time. Oftentimes it is for people who are
considering whether they should put in a nomination for a site. They ask about the process to decide what they
might do and whether it would be worth their while.

Senator URQUHART: So have any improvements to consultation processes been made?

Mr Johnston: Our primary consultation is on behalf of the Australian Heritage Council when they are doing
their formal assessments and particularly the consultations, at the draft stage, which is technically, the 'might have
values' stage. We are also working to do some more consultation early on before the assessment starts. This is 
particularly so with Indigenous communities, where we have looked at it and we have thought that our processes
have not been the best that they could be. So we are doing a lot more of that. We are trying to reach out a lot more
to the communities who have an interest in our assessment periods.

Senator URQUHART: So how do you do that?

Mr Johnston: We use our website. We advertise things and we put advertisements on behalf of the council on
newspapers. We also send officers out to the sites to do consultations. For example—

Senator URQUHART: So you will seek out groups or people or whatever?

Mr Johnston: We try to identify all relevant stakeholders and contact them. There are a couple of formal
processes under the EPBC Act where we have to seek out and write to all interested parties. Often when we are
doing consultation, more parties are identified. We do try and reach everybody.

Mr Oxley: I might put some bounds around a little of what Mr Johnston has said. He is talking principally
about the consultation we do with places that are under formal assessment for inclusion on the National Heritage
List. But we do not run some sort of community outreach roadshow where we go community by community
saying, 'Have you got any heritage you think might be worthy of inclusion on the National Heritage List? Let's
talk about how we might get it there.' We do not have that ability.

Senator URQUHART: Okay. That is all I have. Thank you.

Senator RICE: I will start again with the hooded plovers on the beaches around Warrnambool. We have
hooded plovers, which are an endangered bird species, nest on the beach and horse racing training occurring on
the beach, which is clearly impacting upon the hooded plovers. Last time when I discussed this in estimates, I
think you took some things on notice. I also wrote to the minister and got a reply back from the minister to say
that there was not a required action. The horse training did not constitute a threatening action under the EPBC Act
because it was a series of independent events. So where we got to in our discussion a few hours ago was
discussing whether indeed a series of independent events is, in fact, connected enough to be an action under the
EPBC Act.

Mr Knudson: I think you were also asking about the recovery plan with respect to that species as well. Is that
not correct?

Senator RICE: No. Not the recovery plan for these ones. It is just the status of the horse training and whether
there is in fact a role for the federal government under the EPBC Act to act to protect the plovers.

Mr Knudson: In the first instance, in terms of the status of the species and what sort of referral guidelines
exist on this, I am going to turn to Mr Richardson in a second. With respect to whether something constitutes an
action under the act—and I have to apologise—that is 1.5, where we deal with whether something needs to be
assessed under the act. Mr Richardson can still talk through the specific regulation with respect to how it is listed
under the act and its referral guidelines, I am hoping.

Mr Richardson: Thanks, Senator. The hooded plover is listed, as you said, as vulnerable, I believe. That is
from memory, so I might have to correct myself later.

Senator RICE: It may be in Victoria but in other states—I am not sure—my understanding is that it is
endangered. It may be endangered in Victoria and vulnerable across the country.

Mr Richardson: That is probably right. I have got down here that part of the eastern population or the eastern
extent of the bird is listed. I have also got that there is no recovery plan for this species and that there is not one
required. What it does have is a relatively recent, dating back to 2014, conservation advice, which will set out
essentially what is threatening the bird and why it is eligible for listing as vulnerable, I believe, and then what can
be done about it, if anything. I do not have with me a copy of that conservation advice, though, so I am not going
to talk in much more detail than that. But there is not a recovery plan required for this species.

Senator RICE: Right. If you can table that conservation advice on notice, that would be—

Mr Richardson: That is available online. I am happy to provide it to the committee. But it is available on our
database.

Senator RICE: So now the question is whether indeed the actions of the racehorses are considered to be an
action under the act. Mr Knudson, are you telling me that I cannot ask that question here?

Mr Knudson: Again, my apologies. Whether a particular action is considered to require assessment under the
act is the assessment team. My apologies, Senator. I misunderstood. I thought you wanted to talk about the
recovery plan et cetera for this species, so I fully have to acknowledge that I have misled you earlier. 

Senator RICE: All right. We will park the hooded plovers until 1.5, which is whenever that is later on this
evening.

Mr Knudson: It is the next session.

CHAIR: If you do not have any more questions—

Senator RICE: No, I do. I have quite a few more.

CHAIR: If you hold that one, we will go straight into program 1.5.

Senator RICE: No. I have more on 1.4.

CHAIR: I am saying afterwards. Then we can run that one straight into 1.5.

Senator RICE: Great. Okay. I will move on to core recovery plans and, in particular, the role of recovery
plans and the success of recovery plans so far. To summarise, recovery plans, as Mr Andrews outlined earlier,
contain actions to control threats and to protect habitat, pretty much, do they not?

Mr Knudson: I will ask Mr Richardson to answer.

Mr Richardson: The content of recovery plans is essentially set out in the legislation, and they are close to
what you have described. I guess I would say that, at the highest level, a recovery plan to be made by the minister
or adopted by the minister must provide for the research and management actions necessary to stop the decline
and support the recovery of the listed threatened species so that its chances of long-term survival in nature are
maximised. That is a literal quote from the act. It provides a series of content requirements in the act as well.

Senator RICE: So we do the research and work out what the threats are and then act to address the threats?

Mr Richardson: Indeed. So the recovery plan is essentially a guidance document for a couple of purposes.
One is for those who are looking to invest in recovery actions. The recovery plan must set out, under the
legislation objectives, criteria in order to measure whether you have met those objectives or not. It must set out
the actions needed to pursue or achieve those objectives. It must set out what the threats are that are leading to the
animal being listed.

Senator RICE: How many recovery plans are there?

Mr Richardson: I can give you a rough estimate. I do not know if you need an exact number. It is roughly
about 400. If you need an accurate number, I can take it on notice. That is roughly.

Senator RICE: And how many species have been identified as requiring recovery plans?

Mr Richardson: Currently, there are around 150 species for which recovery plans are required but not
currently in place. I should say that some of those 400 plans will cover more than one species. It is not quite a one
for one.

Senator RICE: We have had recovery plans now for how long?

Mr Richardson: Certainly since the inception of the EPBC Act, so certainly since 2000.

Senator RICE: So we have had recovery plans for 17 years. Focussing on animal species, have we got
instances of recovery plans that have led to animals becoming less threatened?

Mr Richardson: It is a good question and not an easy one to answer. I will explain why that is. Obviously,
there are species that are listed. One way you would measure success, if you like, as we were talking about earlier
for white sharks, is that you would hope that recovery actions are implemented and monitoring led to an
understanding that that species had recovered to the point where it could either be downlisted—so it had come
from critically endangered to vulnerable—or off the list altogether. But the thresholds for those different listing
categories are quite broad, so you can have, if you like, a measured success, but it will not actually lead to
downlisting in the first instance. It might take quite a sustained effort over a very long period of time. Obviously a
lot of these species are listed because of actions or threats that have been acting for a very long period of time or
since European settlement, in a sense, for some of them. So it is not easy to measure. Another point is that there is
a combination of actions taken by state and territory governments, which are the principal land managers where a
lot of these terrestrial species at least reside. So there are actions they are taking that are consistent with our plans
and that our plans provide guidance for. There are also actions that we directly invest in as well from some of our
investment frameworks.

Senator RICE: In terms of which species have recovery plans, do you do an analysis by biodiversity type, so
environment type?

Mr Richardson: As in which ones warrant having a recovery plan? 

Senator RICE: No. In terms of the desert dwelling species, forest dwelling species and marine species. Do
you do that analysis?

Mr Richardson: Until the act was amended in 2007, every listed species was required to have a recovery
plan. It will not surprise you that not all did. We just did not have the resources to do that. We had a lot of
cooperation from states and territory governments to prepare plans. So the vast majority of plans that were in
place by 2007 were plans that states and territories had led the development of and that various ministers have
adopted under the EPBC Act rather than plans we had written ourselves. In 2007, the act was amended to require
all species that were listed to have conservation advices and for the decision to have a recovery plan to be a
decision the minister of the day made on advice from the Threatened Species Scientific Committee. The sorts of, I
guess, considerations that go into the TSSC's advice about whether a recovery plan is warranted or not for a
particular species go to the complexity of the threats acting on the species and the amount of cooperation needed
by other actors—state and territory actors or private sector actors and community groups et cetera. Cross-tenure
species tend to require more cooperation and more leadership, if you like, than a recovery plan will bring because
of the nature of preparing it and the consultation that occurs in its preparation. So it is fair to say that in the last 10
years or so since those amendments, fewer species have been eligible to have a recovery plan than prior to that,
where they all needed it. Given the resources available to prepare these things and the impost on state and
territory governments to input to them, the TSSC is pretty careful about recommending recovery plans on a whim.
They really need to be—

Senator RICE: So are there instances where the TSSC has recommended a recovery plan and no action has
yet been taken to begin the development of it or to fast-track the development of it? I know that there are many
recovery plans that have taken an awful long time in their development.

Mr Richardson: I could not point to examples. I would have to take it on notice to say whether the committee
since 2007 has recommended a recovery plan, the minister has accepted that and we have not yet commenced it. I
think it would be a small minority, if there are those out there. For the bulk of those plans from 2007 onwards,
again, we have been consulting and engaging with the states and territories to commence development. We have
talked about some of those in past estimates and about some difficulties with that. There are some that we have
picked up and finished ourselves. Essentially they end up as plans made by the Commonwealth rather than state
plans adopted by the Commonwealth. But that is time consuming and resource intensive.

Senator RICE: I could not find information on the listed species within forests. You did not seem to identify
which are forest dependent species, as I said, or species from different environments.

Mr Richardson: We tend not to break them up by those categories. I could have a go on notice of giving you
a list of species for which a recovery plan is required that are forest dwelling species.

Senator RICE: That would be very useful, thank you.

Mr Richardson: I will take that on notice.

Senator RICE: In terms of the level of funding for recovery plans, where is that found in the environment
budget? Is that articulated as a line item?

Mr Oxley: It is not articulated as a line item. It is dealt with under program 1.1. Recovery plan objectives and
actions are things that are given consideration when the department is giving advice to the minister about
spending proposals under the National Landcare Program. Threatened species recovery actions are one of the
things that are considered when those grants programs are being allocated.

Senator RICE: So is there any analysis of the resources that are being put in over time into the
implementation of recovery plans?

Mr Oxley: Mr Andrews, the Threatened Species Commissioner, I think has on a reasonably regular basis
provided updates to the Senate and through public communications about the extent of expenditure which is being
made in the service of threatened species recovery objectives. I point back to that information and outcome 1.1.

Senator RICE: What I am getting at is that you have a suite of recovery plans, all of which have a suite of
actions in them. I want to know the level of resources that are being put towards the implementation of those suite
of actions over time.

Mr Oxley: And I am saying to you that the best source of information for that is Mr Andrews, who has done
some of that analysis. It is in program 1.1.

Senator RICE: But does he analyse it according to how much money is actually being spent on recovery
plans overall, or is it just the range of different actions that are being undertaken?

Mr Oxley: We will need to take that on notice. I cannot give you that information now. 

Senator RICE: In doing the analysis—I listened to Mr Andrews talk at various estimates—it is very difficult,
in particular, as you say, if you are interested in the threatened species that are forest dependent, to work out the
resources going to the recovery plans for forest dependent species. That information does not seem to be
available.

Mr Knudson: I am not sure we will be able to do this, but this is what we could aim to try and do. With the
lists that Mr Richardson just talked about, we will try to take a list of what species are forest related and then do a
connection with the spending chart that Mr Andrews has tabled. We will say which of those species that are forest
related are connected to the specific actions for which he has marshalled funding and resources to be spent. It will
not break it down specifically to the specific actions within a recovery plan, but it will give you a relationship
between the specific forest dependent species and the actions that Mr Andrews has been profiling.

Senator RICE: In terms of the quantum of funding that has gone towards those actions, how does that
compare in the forward estimates with previous years?

Mr Oxley: I am sorry, this really is a matter for outcome 1.1. That is where the spending programs are dealt
with.

Mr Knudson: Sorry, Senator. I will get you an update.

Senator RICE: Particularly given that I was told that recovery plans was in 1.4.

CHAIR: If there is any way that we can, without getting officials back, answer Senator Rice's question, that
would be helpful.

Mr Knudson: That is what I was trying to lay out for the senator. To get to the essence, now that I understand
where she is trying to go, it is to understand the amount of spending on forestry dependent species and articulate
what that spending profile looks like. Sorry, Mr Oxley; I interrupted because I talked to the senator previously
about waiting for this outcome. I had understood differently where she wanted to focus. If providing that sort of
analysis would be helpful, that is where we will focus our effort.

Senator RICE: Yes. Thank you. Can I confirm that the Leadbeater's possum recovery plan is still not
complete?

Mr Richardson: That is correct.

Senator RICE: I understand that there was an emergency intervention two years ago in 2015 and that a oneyear
timeline was established for that recovery plan. Is that the case?

Mr Richardson: There was at the time of the threatened species summit, which was held at Melbourne zoo,
and at the same time as the launch of the threatened species strategy. The strategy did list the Leadbeater's possum
as one of the species for emergency action; that is correct.

Senator RICE: There was that emergency intervention with the Leadbeater's possum, and the government
basically said there would be a recovery plan prepared within a year.

Mr Richardson: On an accelerated timeframe; that is correct.

Senator RICE: But within a year was the commitment given at that stage.

Mr Richardson: I think that was in an action plan that was published online that the then minister released;
that is correct.

Senator RICE: I have asked questions about the recovery plan before. It was in draft a year ago—early in
2016—or over a year. Is that the case?

Mr Richardson: About a year; that is correct.

Senator RICE: Is it sitting on the environment minister's desk?

Mr Richardson: It is not. It is sitting on my desk. I will take that one, yes.

Senator RICE: What is the delay in finalising it?

Mr Richardson: It is a couple of things. Firstly, we got an unprecedented number of public comments. I
know that is some time ago. We got more than we were expecting. The person working on that plan essentially
collates all of those comments and documents how they have been taken account of in revising the plan to take
account of the comments. That needs to be documented. It is problematic because there were 3,700 submissions.
That does not sound like a lot but it is a lot from the perspective of a recovery plan and a public comment process.
A small number of them—150-odd—were quite detailed comments going to the science and content of the plan et
cetera. They were taken account of by the drafting group over that plan, which is comprised of members of my
branch; Victorian government DELWP, or Department of Environment, Land, Water and Planning, officers; and 
an independent scientist, Professor John Woinarski. So that group needed to look at all of those comments. They
needed to look at them in detail and agree how the plan was going to be varied to take account of those
comments. Then we needed to document that. Under the legislation, we are required to give all of those
comments to the minister, including a document explaining how they have been taken account of or not and the
reasons why. Whilst only a relatively small proportion of those 3,700 were individual submissions, the balance—

Senator RICE: How many were individual submissions of that 3,700 that you needed to do that level of
documentation?

Mr Richardson: I was going to go to that. I will find it in a moment. The balance of the 3,700 were what I
would term campaign style submissions, but they were individualised, so it meant that you could not treat them
with—

Senator RICE: So the same ask, presumably, was being made in each one, though?

Mr Richardson: No. They were all individually annotated, so they all need to be covered, if I can put it that
way, we think, to meet the requirements of the legislation. So we are working through it and we are close. More
recently, I would also say that there have been a couple of what I would call complicating factors. One is that the
same officers that are doing that job are also responding to a public nomination for a change in the listing status of
the possum. That came through in the process. I am not sure if you were here for it, but Mr Oxley described it
earlier. There was the annual call for public nominations for species to be listed or changed in their listing status.
That is a process that is very time bound in the legislation. We do not have—

Senator RICE: What is the request for a change in the listing status? What is the change that is being asked
for?

Mr Richardson: So it was nominated for downlisting from critically endangered to endangered. That process
now needs to be considered along with all the other public nominations and will be going to the Threatened
Species Scientific Committee in the next meeting, which is in early June—the first week of June, I am pretty sure.
There is then a timeframe set out in the legislation where they need to advise the minister on the proposed priority
assessment list of species that they think warrants reassessment based on the nominations they have received. The
minister has a period after that to determine what ends up on the final priority assessment list. So the officers that
are responsible for the plan are also working on that process as well. We do not have any leeway on time on that.
The statutory time of the recovery plan is April 2018. We are confident we will get in before that, but the other
process needs to be dealt with in the immediate.

Senator RICE: So you are confident that you will get in by April 2018. Do you expect to come in earlier than
that?

Mr Richardson: I would be very hopeful of coming in earlier than that. I note that that is the timeframe for
the plan to be finalised—so for the minister to make a decision on the final plan.

Senator RICE: Is there any interaction between the recovery plan and how it informs or is informed by the
Victorian government's forest taskforce processes?

Mr Richardson: That is another potential complication. I am not saying it is a complication at this point. As
you would be aware, the Victorian government is reviewing their forest prescriptions, or at least their buffers.
There has been a lot in the media about this but also on their website about the review that they are undertaking of
the, I guess, efficacy of the arrangements that were brought in in 2014 out of the Leadbeater's possum advisory
group process that the Victorian government ran. So that is when they introduced their 200-metre buffers around
newly discovered colonies. So they are conducting a review of at least those buffers; I am not sure if it is broader
than that. I understand that their website says that that review will be reported to the government by the end of
April. Presumably, the Victorian government will act on it and make some announcement at some point.

Senator RICE: So they will review it by the end of April. They are now in mid-May. Do you know whether
that has occurred?

Mr Richardson: I do know. I do not believe it has occurred. I have not seen anything. We are keeping an eye
out for it, of course.

Mr Oxley: We have been inquiring and we have not yet received it. There were somewhere north of 50
detailed submissions. So there were 50 plus substantive, detailed submissions.

Mr Knudson: In 2015, a comprehensive conservation advice was prepared for the species. It was approved
back then as well. That has already been taken into account through individual assessments where the species is
potentially impacted.

CHAIR: Senator Rice, how are you going for time? 

Senator RICE: I was just about to respond to that.

CHAIR: I am not saying you need to finish. I am just asking.

Senator RICE: I am almost finished on this. Then I can move on to 1.5. I am sorry, but that—

Mr Knudson: It is conservation advice.

Senator RICE: Was the conservation advice part of the emergency intervention that was put in place in 2015?

Mr Richardson: As I mentioned earlier, at the time of listing, and, in this case uplisting of the species, a new
conservation advice was prepared. So this was done in 2015, as Mr Knudson said, at the time of the uplisting to
critically endangered for the possum. So the then minister made that decision in April 2015 and a new
conservation advice was published.

Senator RICE: And the threatening processes have been identified in the work that you have done on the
recovery plan so far?

Mr Richardson: It has been included in the uplisting as well in the conservation advice that we just referred
to.

Senator RICE: What are those threatening processes?

Mr Richardson: So habitat loss and loss of habitat based on historical impacts. Fire, of course, adds to that.
They are the ones that spring to mind.

Senator RICE: The government has got through its planning process and there is a commitment for animals
not to become further threatened and further endangered. While this time is being taken to develop this recovery
plan, the logging of the Leadbeater's habitat is still occurring.

Mr Richardson: Sorry, what was the question, Senator?

Senator RICE: Well, logging and fire as a result of that logging are still occurring. As a result of logging
operations, post regeneration burns end up burning old hollow bearing trees that have been habitat for the
Leadbeater's possums. It is a big worry given the level of time that this recovery plan has taken to be developed.
The threatening processes are still ongoing.

Mr Knudson: I think that is why I was highlighting the fact that there is that conservation advice. If there are
logging projects outside of an approved regional forest agreement, they would be subject to consideration of that
conservation advice. So, no, we are quite comfortable that we have got advice out to decision-makers to inform
any actions outside of the RFA areas.

Senator RICE: But these are inside the RFA areas where the logging actions are occurring.

Mr Knudson: So if it is logging within the RFA areas, that is subject to that piece of legislation, not the
EPBC Act.

Senator RICE: But once a recovery plan is developed, that will impact on what is able to be done under the
regional forest agreement, will it not?

Mr Knudson: This is why I was flagging that the RFA team will be here for the last part—to deal with
regional forestry agreements. But we can come back, through a question on notice, with the specifics of what
needs to be taken into account under the regional forestry agreement in the central highlands and what its
connection with the recovery plan is. Effectively, it has been approved for a 20-year period. Any changing in the
listing et cetera we can deal with explicitly in that question on notice and with what the implications are.

Senator RICE: Thank you.

CHAIR: Senator Whish-Wilson, you have some more on 1.4?

Senator WHISH-WILSON: Yes. I have a couple of quick questions to Dr Gales. Thank you for getting in
here for us. I have three questions on NEWREP-A after the Japanese whaling fleet went down to the Southern
Ocean this summer. What is the process from here in terms of the scientific working group within the IWC
looking at the scientific justification for that?

Dr Gales: The meeting of the scientific committee has just concluded or is concluding today in Europe. I do
not know the outcomes of that, of course, yet, but they will have been reviewing the submissions from Japan on
both their scientific whaling program in the north Pacific and NEWREP-A in the Southern Ocean. There was
already the expert panel, you will recall, that concluded that Japan had not made a sufficient case to justify the
killing of the whales. At this meeting, we would anticipate that the scientific committee will reach some form of
conclusion, from a range of views, on the expert panel's conclusions. At the last commission meeting, Australia
managed to get agreed a resolution for an intersessional group which I chair. That group's responsibility is to take 
all of these quite technical findings from the scientific committee, from the expert review, and bring them
together into a document that the commission can consider. So we would hope that at the commission meeting
next year, which is biennial—the meeting will not be until October next year—the commission will, perhaps for
the first time, form a much clearer view and make some much clearer statements on their view of Japan's
activities under NEWREP-A and, indeed, their other scientific whaling program.

Senator WHISH-WILSON: I will go to that first before I lose my train of thought. I have a couple of other
quick questions. So that is October 2018, which will precede the whaling fleet going down there again for
summer?

Dr Gales: That is correct.

Senator WHISH-WILSON: If the commission does form a clearer view, as you say, is it going to be still
binding on Japan, or can they still choose to opt out?

Dr Gales: As you may recall, the difficulty with this is that, under the convention itself, article 8 of the
convention provides a right to take whales for the purposes of scientific research. The court case, you will recall,
found their previous program did not satisfy that. Therefore, whaling under JARPA II was illegal. The
commission has no right of veto over Japan, but it is certainly the case that a clearer commentary around the view
of the scientific committee, the view of the expert panel and, one would hope, the view of the commission will
add a great deal of international pressure to Japan on its whaling activities.

Senator WHISH-WILSON: The fisheries ministry in Japan said its research is focussed on the reproductive
and nutritional cycles of minke whales. The purpose of this research is to carry out a detailed calculation of the
catch limit of minke whales and study the structure and dynamics of the ecological system in the Antarctic Ocean.
My understanding is that it was over a period of time—over 12 years—that they wanted to take 4,000 minke
whales. Are you confident that the committee will be able to form a view from just one season of whale slaughter,
or do we have to wait for another 11, do you think?

Dr Gales: Well, the expert panel formed a view on the proposal that Japan put and the scientific justification
for their taking of whales. Their view was that Japan had not made a sufficient scientific case to justify the killing
of whales. I cannot speak for the conclusions the scientific committee will draw, but I would hope that the
scientific committee considers those very carefully and forms its own view on the basis of the same evidence that
was presented to the expert panel. Australian scientists are at the meeting at the moment and are certainly pressing
the view that Australia's view very strongly is that Japan has certainly not made a case to justify the killing of
whales. Australia and many other countries have demonstrated that the science can be done with nonlethal
techniques and, indeed, done better with nonlethal techniques.

Senator WHISH-WILSON: Okay. This is the last question for me. I am very hopeful, based on what you
said, that maybe we will get a clear statement. Is there any other process or intergovernmental process going on at
the moment in relation to potential court action or other representations to the Japanese government on this issue,
or are you waiting on today's outcome and potentially a statement by 2018?

Dr Gales: Australia's view is certainly that our most effective—it is not a simple task, as you know—
Senator WHISH-WILSON: No. I understand.

Dr Gales: That the most effective action is to exert pressure through the formal processes in the International
Whaling Commission. Australia will continue to do that. Australia also engages diplomatically at other levels,
where appropriate. As you know, the whaling issue is isolated from our broader, otherwise very positive
interaction with Japan on most issues. But, having said that, outside of the IWC, Australia's views on whaling
from our minister and, indeed, our Prime Minister have been many times made clear to Japan. So Australia does
exert pressure in other fora, but the primary engagement is through the International Whaling Commission.

Senator WHISH-WILSON: I want to ask you very quickly about the Prime Minister making that clear. Are
you referring to the current Prime Minister, Malcolm Turnbull, or are you referring to previous Prime Ministers?
Dr Gales: Well, both. And certainly our current Prime Minister has made his views very clear to the Japanese
Prime Minister as well.

Senator WHISH-WILSON: I just cannot remember Tony Abbott ever doing that; that is all. But I will take
your word for it. Thank you.

CHAIR: Thank you very much.

Senator RICE: My question is in 1.4.

CHAIR: You had better make it snappy. 

Senator RICE: Okay. In terms of the information that has been taken on notice for me, and the forest
dwelling species which are listed which have got recovery plans, in forests that are subject to logging, my
understanding is that the main mechanisms to be protecting those species are implemented through the regional
forest agreements. That is correct, is it not?

Mr Knudson: That is correct. But in each of those regional forestry agreements, the state that is the manager
of those will have their own management plans and actions that they will put in place to ensure that the RFA has
that appropriate balance between ecological outcomes and forestry.

Senator RICE: As well as the current status, what was the status of those forest dwelling species when the
regional forest agreements were first brought into being 20 years ago?

Mr Knudson: I am not sure how we would answer that because the EPBC Act would not have existed at that
point, if I am correct in terms of the map.

Mr Oxley: Threatened species listing legislation predates the EPBC Act. All those listed species were
grandfathered into the EPBC Act. So hopefully we can readily mine the data holdings we have to produce that
information and come back to you on notice.

Senator Ruston: Senator Rice, I am sorry that I was not here when Gregory Andrews would have been here
earlier in the day. My understanding is that Dr Andrews was very clear that no species has been made extinct or
has been increased on the list as a result of logging. There have been other reasons that have required an
escalation of listing. So I just put that on the record because you seem to be—you may not be, and I will not
verbal you—making a correlation between logging and species extinction.

Senator RICE: I am indeed. As we have discussed, you are addressing the threats in the recovery plans. The
threats are basically, in the case of these animals, loss of habitat and predation, essentially. Clearly, for these
forest dwelling species, there is enormous evidence that intensive logging reduces the habitat that is available.

CHAIR: Senator Rice, I am not quite sure this is a question.

Senator Ruston: Senator Rice and I will have a fun time in the actual review of it. I just draw it to your
attention—

CHAIR: Thank you, Minister. Just on this, that was my recollection as well. The Threatened Species
Commissioner said—we can get the details because it is in the Hansard—

Senator RICE: The information that is going to be provided for me on notice about the status of the various
forest dwelling animals is going to be drilling down to some meaningful information. Dr Andrews talks a lot
about a range of species. He does not talk about forest dwelling species.

CHAIR: I think if you are satisfied with the information that they have taken on notice—

Senator RICE: I just want to clarify one more thing.

Senator Ruston: When you are finished, I would also like to clarify an inaccurate statement made by Senator
Rice that I have just received clarification on.

CHAIR: Please, Minister.

Senator Ruston: You made a comment before that species had been made further extinct or further
endangered on the basis of post harvest logging burnback. I have just been advised that that is absolutely not
correct.

Senator RICE: In fact, I think I have tabled at estimates before examples where you have had trees which
have had—

CHAIR: Senator Rice, I do not want to turn this into a debating point between you and the minister. Are there
any other questions? If you can phrase it as a question, I think that would be helpful.

Senator RICE: Well, my question is: have you had brought to your attention the examples that have been
brought to my attention of many hollow bearing trees which have been killed and many which have subsequently
also blown down because of post harvest burning activities?

Senator Ruston: No. In fact, I have been advised of the introduction of a new method to deal with post
logging. The variable habitat model actually results in trees with hollows not being burned. It is there for the
express purpose of dealing with the issue that you are talking about.

Senator RICE: Is that—

Senator Ruston: Is it okay if I finish, or are you just going to talk over me? My advice is that in response to
the protection plans put in place because of an identified problem with the Leadbeater's possum, a number of
actions have been put in place that are actually seeing the recovery of this particular species. The actions taken in
the past that were brought to our attention are no longer happening. So I think it is a positive story about the
recovery plan for the Leadbeater's possum.

CHAIR: Any other questions, Senator Rice?

Senator RICE: I did have one further question. I want to clarify that the policy for forest management under
which logging still occurs is the national forest policy from 1992? That is still the basic forest policy?

Senator Ruston: In substance it is, yes. It has been amended subsequently, but that is still the basis on which
our policy development in forestry occurs, yes.

Senator RICE: So the goals and objectives outlined in that national forest policy are still the ones we are
aiming to meet in our management of forests?

Senator Ruston: Well, subject, obviously, to the overlaying of RFAs that have been subsequently developed.

Senator RICE: Well, the RFAs essentially aim to be the implementation mechanisms of the national forest
policy.

Senator Ruston: And we are in the process of making changes to those, as you are well aware at the moment,
to try to reflect current day practice.

CHAIR: Thank you, Senator Rice. Senator Whish-Wilson, you have one final question on 1.4?

Senator WHISH-WILSON: This is a question for Dr Gales. There have been some research papers released
in the last six months around population levels for humpback whales recovery. There was a suggestion in one of
them that perhaps they may be delisted as a protected species. Have you got any comments on the humpback
whale population listing? Have you done any work on that issue?

Dr Gales: Certainly the trend for the two separate populations on both the east and the west coasts of
Australia is that they are both increasing. They have been since the early 1960s from a very, very low base. They
have been increasing at least on the east coast, where we have very good data. They have been increasing at close
to their maximum biological rate, which is a really encouraging element. So the population off the east coast is
probably now numbering close to 30,000. Our data on the west coast is a little out of date. There is some
discussion I am aware of at the scientific committee this year seeking further data on the west coast population.
But it has generally been slightly larger than the east coast. There have been some international discussions. In
fact, in the US, some populations of humpbacks have been delisted. I am not aware—and other colleagues in the
department could talk to this—of any proposals in Australia.

Mr Richardson: It has not received a nomination, so it has not been considered to date by the Threatened
Species Scientific Committee for adding to the peak FPAL process. You mentioned it being delisted as a
protected species. Just to correct you, it was going to be delisted as a threatened species.

Senator WHISH-WILSON: A threatened species, yes.

Mr Richardson: It would still remain a listed migratory species under the convention on migratory species, so
it would still receive statutory protection. That has not been commenced yet.

Senator WHISH-WILSON: Firstly—you can take it on notice if you do not know, Dr Gales—do we know
roughly what their population levels were previously before they were put under intense pressure?

Dr Gales: We are aware that towards the end of the 1950s and into the early 1960s, both populations were
down at a level of a few hundred animals. So they were precipitously low. It turned out later that a lot of that final
decline to incredibly low numbers was attributable to illegal Soviet Union whaling in the area. They took quite
large numbers of humpbacks. So they came from an incredibly low base.

Senator WHISH-WILSON: So it was pressure from whaling that got them down to that level?

Dr Gales: That is correct.

Senator WHISH-WILSON: Given that they migrate to the Southern Ocean as well, with similar patterns to
some of the other species, such as the southern right whales, would you have any concern that they may be
targeted by Japanese whalers?

Dr Gales: There is certainly no indication that Japan intends to target any species in the Southern Ocean other
than the minke whales.

Senator WHISH-WILSON: Thank you.

CHAIR: Thank you very much. Mr Richardson, you have a point of clarification? 

Mr Richardson: I think Senator Lines, just before she left, asked a question about a letter from the WA
minister, on shark research. We were not sure of the answer. We have had it clarified. We understand that the
response to that letter from the minister has gone back.

CHAIR: Thank you very much. We will make sure that information gets to Senator Lines. Thank you. That
now concludes program 1.4. Happily, because we are so ahead of the program today, we have had agreement
amongst the committee members, and I understand with the committee and Broadcasting, that we will go through
until the conclusion of 1.5. That will take us into the dinner break, but probably by only about 30 minutes or so.
That way, we can conclude and we do not have to come back after a dinner break.

[18:01]

CHAIR: Subject to everybody's concurrence with that, we now move to program 1.5, environmental
regulation.

Senator CHISHOLM: I will start with questions on tree clearing in Queensland. I thought I would come back
to the appropriate body. I learned my lesson in February. In February, when asked, you said that under federal
law, the EPBC Act applies. If there is a significant impact on the Reef which is a matter of national environmental
significance, then the EPBC Act is in play. So regardless, frankly, of Queensland law, the federal law will apply.
Has the federal government taken any action under the EPBC Act on tree clearing that has taken place in the
Great Barrier Reef catchment area?

Dr de Brouwer: Mr Knudson started answering some of that question and went through the 54 or so permit
holders that the department is engaged with. We might just go through where the progress is on those permit
holders.

Mr Knudson: If you do not mind, I will turn to my colleague Mr Cahill to walk through where things are up
to on that.

Mr Cahill: There were 59 permits granted to 54 property holders that we decided to make inquiries with.
Through that process, we both not only looked at the matters but engaged with them directly as well as with the
support of AgForce and the National Farmers' Federation. Mr Knudson this morning confirmed that 46 property
owners have now been advised that their proposed or completed clearance is unlikely to have a significant impact
and so no further action is warranted. Six property owners have been advised that approval is likely to be referred
for some or all of their proposed clearing. There are two that we have been looking at more closely. I can update
you further on those specific six and two, so I will just pull that information out for you.

Of those six properties likely to be referred, two have now been referred to us and are going through an
assessment process. As part of that exercise, if a matter is being referred, they cannot proceed with any action
until there is a decision on that referral. One has indicated that clearing will not proceed due to a change in
circumstances in terms of their business. One property was deemed to be referred, and that was Kingvale. They
have not proceeded with any clearing while they engage their own ecologists and do more work. There are two
more expected to be referred. That is the six. Of the two that were outstanding, we have had an engagement with
one property. We have advised them that if they plan to do any more clearing, we will have to look at that more
closely. There is one outstanding compliance matter.

Senator CHISHOLM: Has the department been engaged with the communities and/or the state government
in a public awareness campaign on the impact that tree clearing in these areas can have on the Reef?

Mr Cahill: There are two approaches we take. Firstly, we have been in conversation with the Queensland
government and using that as a lesson for New South Wales changes to ensure that, when state laws change, there
is a clear communication about what the implications are for the Environmental Protection and Biodiversity
Conservation Act. Secondly, we have been working with the industry associations as good vehicles to be able to
communicate with local communities. We are very mindful that farming and local communities will get inundated
with a lot of information from a lot of different sources, so we are looking to leverage that through the traditional
sources of who they get their information from, such as AgForce and NFF.

Senator CHISHOLM: Do you have any data about how much land has been cleared in the Barrier Reef
catchment area?

Mr Cahill: I do not definitively. We have what has been subject to referral or anything that we have been
looking at. But definitively and more broadly, I would have to take that on notice.

Senator RICE: We have racehorses being trained on beaches in Warrnambool. This is occurring regularly. It
is identified by the trainers as a strategy for training their horses. They are impacting upon the plovers. I wrote to
the minister and was told that this was not an action that could be addressed under the EPBC Act because they 
were individual instances and they were not connected up in a series of actions. I want to explore whether these
actions are being considered as an action under the EPBC Act.

Ms Collins: I understand that the activity you are referring to is in the Belfast reserve in Victoria.
Senator RICE: Yes.

Ms Collins: Primarily, the management of public land is a matter for the state government—in this case, the
Victorian government. The Australian government has a role if there is a significant impact on matters of national
environmental significance which are identified under the Environment Protection and Biodiversity Conservation
Act. So an individual act, as you have pointed out, can only be considered under the act if it constitutes a
significant impact. I do understand that the Threatened Species Commissioner has been engaging on this. He has
consulted with Birdlife Australia and Friends of the Hooded Plover and has engaged with the authorities to offer
to assist. I understand that Parks Victoria is developing a plan of management for the Belfast Coastal Reserve and
will be considering the management of the beach as part of that plan of management.

Senator RICE: With regard to the federal government's role and whether the horse training is a significant
action under the EPBC Act such that they would have to seek approval of the federal government for that to
continue, what is the current status of that?

Ms Collins: My understanding is that if it is an organised activity—for example, one horse trainer undertaking
routine horse training activities on the beach with a number of individuals—you could look at it as an individual
and then assess the cumulative impact. But if it is not being coordinated or undertaken by a single operator at any
level, it is seen as individual actions. The EPBC Act is triggered when an action is having a significant impact on
a species.

Senator RICE: So in terms of it being organised, you have trainers that are training their horses there on a
regular basis and more than one horse. So is that enough for it to be a significant action?

Mr Knudson: I think what Ms Collins is talking about is that under the act you need a proponent. You need
someone to say, 'All right. I have organised these seven different trainers to come and train their horses and I am
accountable.' Therefore, it would be subject to the act, if I have understood it correctly.

Ms Collins: That is right, yes.

Senator RICE: So you would need to have more than one trainer, are you saying?

Mr Knudson: But organised and run. So if you had a company that was running a business where they had
seven different trainers operating in a company on the beach, you would say, yes, that company owner is the
proponent for this action.

Senator RICE: But if you just had one trainer that was undertaking a series of horse training activities and
which were the equivalent of many horses on the beach, surely that would be sufficient to be a series of connected
activities to be a significant action.

Mr Tregurtha: You are absolutely right. Each individual trainer is obligated under the EPBC Act to assess
their own activity as to whether or not it constitutes a significant impact under the EPBC Act. We provide
significant impact guidelines to assist people to make that self-assessment. We have a compliance and
enforcement function that looks at individual instances where they are referred to us. But each entity undertaking
their own action needs to consider whether or not they are having a significant impact undertaking the particular
activity they are doing. It does not consider everything else that is going on. Each individual instance needs to be
considered by each proponent, if you like.

Senator RICE: So have these horse trainers done that self-assessment?

Mr Tregurtha: I am unaware. There is an obligation under Commonwealth law for those self-assessments to
be—

Senator RICE: What if the department is made aware of the fact that there is a strong case that they should be
doing that self-assessment and that has not occurred?

Mr Cahill: So, practically, if you have lots of trainers, how do you get a picture of what is happening overall?
We will talk to Parks Victoria, because, in doing their proposed management plan, they should be standing back
and looking at what is happening on that reserve. We might have a strategic engagement with Parks Victoria to
see about what their approach is for managing that reserve and what are the implications for our actions.

Senator RICE: But would Parks Victoria do an assessment for you as to whether there is a significant action
occurring and there should be an assessment undertaken? 

Mr Cahill: What I would be looking for from Parks Victoria is what information they are taking into
consideration if they are looking to do a management plan. We would look at the material from the Friends of the
Hooded Plover and the Threatened Species Commissioner. We would stand back and look at that holistically and
say, 'Okay, what does that mean for the EPBC Act?' I am hoping that there will be some good information from
Parks Victoria. We would like to stand back and see what information is at hand, particularly when you have
Parks Victoria managing that reserve and you have a range of local communities. Let us bring that information to
the surface and look at what that means and the implication for the EPBC Act.

Senator RICE: I want to clarify that you have already requested Parks Victoria?

Mr Cahill: No. We have not. Not to my knowledge. But I will undertake to have an engagement with Parks
Victoria to see what information is at hand and see if we can find a sensible way for us to be able to manage what
that means for a species that we protect.

Senator RICE: Would you be able to give me an idea of a timeline that that engagement could occur under?

Mr Cahill: No. I will have to take that on notice. But I will give an undertaking to contact them this week.

Senator RICE: Thank you. In particular, I understand that there was a recent Federal Court decision in the
Tasmanian Aboriginal Centre v Secretary of the Department of Primary Industries, Parks, Water and the
Environment that reaffirmed the meaning of an action to include a connected series of smaller activities or
instances of conduct that form a greater whole. So it seems to me that there is a very strong case that these smaller
activities forming a greater whole would form an action that would need to be assessed under the EPBC Act.

Mr Cahill: I think in the case that you refer to, again, the benefit of that is there is a level of government
involved in pulling together that picture. It makes it more practical for us to stand back and say, 'What is the
Tasmanian government looking at there?' We can have a similar engagement with the Victorian government and
try to understand what they know about it, what their intent is in managing it, what the implications are for a
species we protect and what is the most practical way for us to apply the EPBC Act.

Senator RICE: Thank you.

Senator WILLIAMS: Who is the expert on sharks?

Mr Cahill: We have a few.

Senator WILLIAMS: Who could take me through the New South Wales shark management plan and how
effective it has been?

Ms Farrant: Did you have a particular question in mind around that shark management plan?

Senator WILLIAMS: Can you give us a briefing of the general plan of the state, please?

Ms Farrant: The shark management plan that New South Wales provided to the Commonwealth effectively
forms the conditions upon which the national interest exemption is granted by the minister. So it sets out the
constraints. It sets out mitigation factors that the New South Wales government will be using in terms of its shark
netting trial.

Senator WILLIAMS: Have they been using the smart drum lines and nets?

Ms Farrant: Yes, they have.

Senator WILLIAMS: How effective are those?

Ms Farrant: I need to clarify this. The smart drum line trial is separate from—

Senator WILLIAMS: From the nets?

Ms Farrant: From the nets.

Senator WILLIAMS: Exactly.

Ms Farrant: And the national interest exemption.

Senator WILLIAMS: How effective have they been, do you know?

Ms Farrant: The smart drum lines?

Senator WILLIAMS: Yes.

Ms Farrant: I believe from the reports and our engagement with New South Wales that they have been very
effective as a nonlethal mechanism for deterring the sharks. I believe there has only been one shark mortality
from the use of those smart drum lines, and that was because the shark became entangled in the drum line. Aside
from that, it has been—

Senator WILLIAMS: Have any large white sharks been caught and released, do you know? 

Mr Cahill: The trial on the New South Wales north coast for shark netting involved the deployment of up to
10 mesh nets trialled on a certain number of beaches in northern New South Wales. Part of that trial included that
they would do an evaluation against a range of objectives of the trial itself. That included comparing the
effectiveness of the nets with smart drum lines. They wanted to make a comparison between catching dangerous
sharks while minimising impacts on fauna. They wanted to test new devices or procedures which might deter
crustaceans from becoming entangled in the nets and alert researchers in real time of large animal entanglements.
They wanted to monitor the local community acceptance of the presence and operation of those nets during the
trial. In New South Wales, they have the results of that trial so far. They are looking to collate those results and
provide a public report on how that trial progressed against those three objectives. That is not available yet. We
will look at that closely once it has been released.

Senator WILLIAMS: This might be a silly question, Mr Cahill, but does the department put human life
above animal life? The reason I ask is we had two witnesses at a recent hearing at Byron Bay during the day who
said that animal life is equal to human life, which I found quite strange.

Mr Cahill: The best way to answer that question is that the minister—and he has put it on the record as well
himself—

Senator Ruston: Do you want me to answer that?

Senator WILLIAMS: That would be lovely, thank you, Minister.

Senator Ruston: The government absolutely rates public safety as being of paramount importance above all
else.

Senator WILLIAMS: We had these two witnesses, Minister. I thought, 'If I were driving a B-double down a
steep hill and a little boy or girl ran out on the left side and a kangaroo came out on the right, I know which one I
would swerve to hit and which one I would save.' But two separate witnesses that day said, 'No, the life of an
animal is equal to that of a human.' I will be putting my additional reports in on that hearing. Thank you, Chair.

Mr Knudson: I will add just very quickly that we have the stats we just talked about regarding the increase in
the shark smart drum lines in northern New South Wales. One shark died on the line. A number were towed out
and released. Also important—the minister very clearly took this into account in his decision-making on this—is
the impact on human life. There has not been an attack on the beaches where those drum lines have been
deployed. I think we cannot necessarily establish causality, but there is definitely a correlation.
Senator WILLIAMS: It is good stuff.

CHAIR: I want to pick this point up further. We are hoping to have another hearing in Perth to get some of
this anecdotal evidence and information that has been put to me. In the absence of definitive research or evidence,
it is a little difficult to argue. You had some arguments—and, again, they are reflected here today—that the
absence of deaths actually proves that they are ineffective because they have not been caught by sharks. Some of
us look at it the other way around. It is proof that they are working. On that basis, it has been put to me, and I
think others on that inquiry, that the fishermen and the cray fishermen and others say that there has been a
behavioural change on the part of the white sharks. In the absence of drum lines and nets, as an apex predator,
they are no longer afraid of humans. What they are saying, anecdotally, is that the behaviour they are witnessing
every day out in the ocean with the great whites is changing. They are becoming much more aggressive. They are
not showing any fear any more. I do not know of any research. I know that CSIRO is doing a count of the number
of sharks. In terms of the outcomes and assessment of the New South Wales trial or the Queensland trial, has
there been any evidence of or has anyone looked for any behavioural changes as perhaps a cause? It is not just to
catch them but also to deter them.

Mr Cahill: I understand it was a Sydney senator in the hearing in Sydney. We listened to the evidence
carefully. I am quite mindful that New South Wales are taking a reasonably comprehensive approach to looking at
all the information from research through to what they experienced in the trial and community engagement. So we
will look quite keenly to see what the results of what that are, and what that reveals and comments like that.

CHAIR: So you think that, given their comprehensiveness, they could be looking at behaviour in some detail?
Mr Knudson: We talked about the CSIRO study on population estimates.

CHAIR: Yes.

Mr Knudson: There are six other studies. As I seem to recall, it is looking at different shark attack mitigation
technologies and their effectiveness. That is looking at the behaviour of the sharks in terms of not their level of
aggressiveness but how they respond to different technologies being deployed.

CHAIR: Thank you. Just on that, if there is any need to take this on notice and if there is any additional
information—you have already got that from New South Wales, for example—I would be grateful. What has also
been said to the inquiry in evidence is that the behaviour of the sharks is changing. It now varies. There is
variance. So a great white is not just a great white. The different populations of great white are also exhibiting
different behaviours. You obviously cannot answer that here, but you could take that on notice.

Mr Cahill: I will take that on notice, including the scope and depth of the New South Wales evaluation.
CHAIR: Thank you. I am grateful.

Senator URQUHART: I want to follow on from Senator Chisholm with one question about tree clearing. Has
the department been consulting with environmental groups on the tree clearing EPBC actions?
Mr Cahill: In which regards? In Queensland?

Senator URQUHART: Yes.

Mr Cahill: We have not been directly consulting, but we do regularly get information put forward to us
through our office of compliance from environmental and other stakeholders about matters of concern.

Senator URQUHART: What about in other areas? Do you consult at all? Do you talk to groups like the
National Farmers' Federation?

Mr Cahill: We have regular engagements with a range of stakeholders—the NFF, AgForce, the
Environmental Defenders Offices and a range of stakeholders. That is not only individually. We also bring them
together to have discussions about how we apply environmental law.

Senator URQUHART: I want to ask some questions on the fracking study. Can the department provide detail
regarding the research in the budget for water and gas, and the funding? I understand that it allocates $28.7
million to develop an east coast gas development program. Can you give me some detail on what that funding is
for?

Dr de Brouwer: There are two elements in the budget papers around gas. One is the money that has gone to
industry for around that sum. I will have to get my budget papers. That has gone to industry for geoscience.
Senator URQUHART: So $28.7 million?

Dr de Brouwer: There are two major programs. The $28.7 million is for the industry portfolio. That is really
work for geoscience to assess gas. The other is, though, $30.4 million for bioregional assessments, which is
looking at the interaction of gas reserves or coal seam gas with water.

Senator URQUHART: So $28.7 million for industry for geoscience?

Dr de Brouwer: Yes.

Senator URQUHART: And $34.5 million, was it?

Mr Knudson: It is $30.4 million, I believe.

Dr de Brouwer: It is $30.4 million.

Senator URQUHART: It is $30.4 million, sorry.

Dr de Brouwer: So the industry study is around gas and exploration development, but it really should go to
industry. We can talk about that tomorrow on energy in 4.1.

Senator URQUHART: Okay.

Dr de Brouwer: We can go through it in a bit more detail there.

Senator URQUHART: That is right. If I want a bit more detail, I can—

Dr de Brouwer: That is tomorrow, 4.1.

Senator URQUHART: Has the department reviewed state policies into CSG production and fracking?

Dr de Brouwer: That is really an issue around the energy topic, the 4.1. I think you would be looking at the
regulation of CSG or government policy around that.

Senator URQUHART: I want to know if you have reviewed the state policies and, if so, what the findings
are around that. Have you done any of that work? Has the department done any of that work?

Dr de Brouwer: Are you referring to the moratoriums around coal seam gas?

Senator URQUHART: Well, the moratoriums, but particularly in relation to what policies the states might
have into the production and fracking of coal seam gas? 

Dr de Brouwer: It is probably better to talk in item 4.1 around that. Our direct take on this one and the
environment department generally is around the interactions of coal seam gas with water and those systems. That
was in the information—

Senator URQUHART: Have you done any work there?

Dr de Brouwer: We do a lot. In that respect, there are extensive bioregional studies as well as support for the
independent expert scientific committee, which provides advice around coal seam gas and water. In the energy
area, there is a lot of discussion around the impact of state policy, particularly moratoriums and bans on the
supply of gas. That industry measure that we mentioned—supporting the development of new onshore gas supply,
the $28.7 million program—is designed to provide an incentive for onshore gas exploration and development. But
that is, again, provided through the industry department. Those issues are going to come up in the array of policies
that the government has got in increasing the gas supply domestically and the efficiency of the domestic gas
market. But 4.1 is the best place to raise that.

Senator URQUHART: I will follow that up in 4.1. In relation to Abbot Point, it was reported yesterday that
the government has revoked a number of conditions from the Abbot Point approval and replaced them with one
condition to provide the Reef Trust with $450,000. Can you confirm that the reports about conditions being
removed are correct?

Dr de Brouwer: They are incorrect. We can explain that.

Mr Knudson: I will turn to my colleagues if they feel they want to add anything more to this. What has
happened here is this is the exact same amount of money being required for the exact same period of time and for
the same purpose. We have changed the conditions to allow that funding to go through the Reef Trust. The Reef
Trust in turn will then develop the projects with the advice of the independent expert panel that has been set up
under the Reef 2050 Plan to make sure that those projects are as scientifically robust as possible, are focused on
the key threats in and around the reef and that address the issues that are most important. Whereas under a regular
project approval, a proponent could come in and say, 'I'm going to spend $450,000 on this matter,' but it is not
done in a linked up way and an integrated way as the Reef Trust allows. That is what has changed in these
conditions.

Senator URQUHART: The conditions referred to a turtle plan and a marine plan. Have those plans been
done?

Ms Collins: No, those plans have not been done and the project has not yet commenced. But the condition that
has been replaced in the variation will enable the Reef Trust to be able to offset exactly the same measures as
were outlined in the original set of conditions related to the approval.

Senator URQUHART: Does that actually take out a layer of ministerial approval of the plans?

Ms Collins: The Reef Trust is a government endorsed policy position and it was set up to implement the 2050
reef sustainability plan and that plan comprises measures for the protection of things like the sea turtles and the
World Heritage values of the Great Barrier Reef Marine Park. In the offset provision now being made into a
payment to the trust, the trust will be able to enable the exact same offsets but just in that more coordinated way,
informed by the best available science.

Senator URQUHART: Why hasn't the turtle plan and the marine plan been done? You said the plans have
not been done but there was a variation.

Mr Cahill: The vehicle for delivering the offset is now the Reef Trust. The Reef Trust has its own governance
arrangements in terms of ensuring that the outcomes it is trying to produce are governed. They go through an
expert committee. They will do consultation and a range of other things. Through that vehicle, they will produce
the plans to ensure that the offsets are delivered. The requirement is that they are only to be put in place before a
commencement of any mining. Our expectation is that there are superior outcomes from this approach not only
because of the better targeting; there is an independent advisory panel or other arrangements to ensure that the
quality of the offsets delivered by the Reef Trust are in a much better condition.

Senator URQUHART: I understood that previously the minister was to approve the plans. Does this
condition change mean that they will not? So does the trust then seek ministerial approval?

Mr Cahill: Any revision of a management plan or anything will be reviewed by the department and would not
necessarily be informed to the minister or his or her delegate.

Senator URQUHART: But not approval, as was previously the case. You just said 'informed'. I am just
trying to clarify that.

Mr Cahill: I would have to take that on notice to be accurate.

Senator URQUHART: Why is the Reef Trust doing this and not Adani? What is the nature of the offsets?

Mr Knudson: We talked about that a fair amount when Ms Parry was at the table. This is the $210 million
fund that has been set up. Most recently, its phase 4 and phase 5 investments have been in dealing with nutrient
run-off, gully erosion and a range of other activities in and around the reef. They are looking at those onshore and
marine environment investments. So they have built up a fair amount of experience over the last several years in
designing projects to deliver the types of environmental outcomes that we are looking for. This is absolutely a
good news story that these offsets will be going through the Reef Trust.

Senator URQUHART: The condition 37 states that construction must not commence until the marine offsets
strategy has been approved by the minister in writing. The approved marine offsets strategy must be implemented.
So has the minister approved this plan and the turtle plan?

Ms Collins: No. That plan has not been prepared and it has not been approved by the minister. Because the
program has now translated to the Reef Trust that is no longer necessary under this individual approval, but it is
taken over by the requirements of the Reef Trust itself.

Senator URQUHART: The question of whether the Reef Trust makes the decision and then the department
advises the minister—am I clear on that?

Mr Cahill: I will have to take that on notice.

Senator URQUHART: I have some questions around the EPBC Act and third-party appeals. What is the
government's current view on section 487 of the EPBC Act as it is at present in regard to allowing standing third
parties to request court reviews in relation to the federal government environment decisions?

Dr de Brouwer: As department officials, the government's policy stands as it was previously. That was a
cabinet decision. Until there is a reconsideration of that then that policy stands—that is the abolition of section
487.

Senator URQUHART: Is the department currently working on or preparing legislation to amend section 487
of the act in order to restrict the types of groups that would have standing to appeal decisions under the act?

Dr de Brouwer: That legislation lapsed when parliament was prorogued and it is a government decision as to
whether that is brought back or not. We do not have any government decision.

Senator URQUHART: You are not currently working on that?

Dr de Brouwer: No.

Senator URQUHART: Is the department, within the term of this parliament, planning to work on or prepare
legislation to amend section 487 of the act in order to restrict the types of grounds that would have standing to
appeal decisions under the act?

Dr de Brouwer: That is really a matter for the government to consider.

Senator URQUHART: You are not preparing legislation or anything on that at this stage. Does that go back
to the cabinet position that you were talking about?

Dr de Brouwer: Yes.

Senator URQUHART: Minister, I asked whether the department, within the term of this parliament, is
planning to work on or prepare legislation to amend section 487 of the EPBC Act in order to restrict the types of
groups that would have standing to appeal decisions under the act.

Senator Ruston: I will have to take that on notice and take it up with the minister.

Senator WATERS: I want to go back to the Adani conditions issue. Thank you for your earlier clarification,
but I am afraid I have a number of follow-up questions. I do not feel in any way persuaded that there has been an
improvement, so I have some detailed questions that hopefully you can help me with. The previous requirement
was a legally binding requirement because it was in the conditions of approval. My understanding is that the Reef
Trust guidelines do not have such a status. That is my first question to you: how is it an improvement? I think you
said it was an absolute improvement, when they are actually not legally binding anymore.

Mr Knudson: On the question of the specific level of binding, I will have to take that on notice because Ms
Parry oversees the Reef Trust and would know all of the administrative specificities there. I just do not want to
mislead you. That being said, I said it was an improvement because the investment decisions made by the Reef
Trust—given their history of being involved in the reef and delivering projects for good environmental outcomes,
and the fact that it has existed for a while—are informed directly by the Independent Expert Panel. We talked
earlier in this session about their thoughts on 'where to' on the reef. The fact that there is that oversight body of
experts that are providing advice on where best to make those investments, I would argue, is a far better 
opportunity than an individual company trying, with their best intent, to figure out what would be most effective
for delivering outcomes for the reef. That is why I said that.

Senator WATERS: With respect, it is not the company that just tries to figure it out on their own. They put a
proposal to the minister, and the minister has to tick it off. That will no longer occur because it is going through
this different Reef Trust process. So my question is: could the independent expert body have been involved in that
initial drafting of the marine offset strategy under the normal process? You are trying to say that it is somehow
better because the independent mob can get involved this time around. What was stopping them getting involved
the first time, anyway, and still having it be legally binding?

Mr Knudson: Again, not knowing the specific terms of reference for the Independent Expert Panel, I do not
know whether they could have or could not have. But that is easy for us to come back to you on.
Senator WATERS: Well, I think they could have, because the minister can seek advice from anyone he likes
in the course of the standard EPBC approval process.

Dr de Brouwer: Probably, the answer also lies in that it is a more holistic process. The Independent Expert
Panel would provide its advice in the context of a range of different proposals or possible expenditures. It would
be providing that assessment in that broader context. It is an efficient process; it is less costly in terms of the
process. That particular plan or idea would be then put in the context of a range of other proposals at that time and
could be balanced around that, so it is more strategic. I think that is the basis.

Senator WATERS: I understand you, but my point is that they could have done that anyway.

Dr de Brouwer: Again, in terms of administrative simplicity, how do you bring them in, how do you fit? It is
a more streamlined way of doing it and it also puts all of the information together at the same time so that they
can make a scientifically based strategic call on the different forms of expenditure.

Senator WATERS: Yes, but the fact remains that you have taken nine legally-binding conditions from the
original approval. You have deleted those. You have replaced them with one condition that hands the whole
decision over to the Reef Trust. Presumably, that you have taken it on notice means that there is less involvement
of the minister. You cannot even tell me whether the minister still has to tick off on the plan, which is not legally
binding. I am sure that if the minister does not get a say the community is not going to get a chance at all to
review that plan. On those three parameters, it seems a weakening to me.

Ms Collins: Part of the establishment of the Reef Trust is that it is specifically able to accept and implement
offsets from developments that are approved under the EPBC Act. In fact, this is not the first time that the Reef
Trust has been engaged to provide those offsets. Since its establishment, the Reef Trust has provided more than
$5 million worth of offsets.

Senator WATERS: Yes, but that is my very point. You can still use the Reef Trust mechanism but have the
legally binding requirement in the permit. Why, in this case, has it been taken out of the permit and just given to
the Reef Trust with much less scrutiny, accountability and structure around it? In all of those other instances, this
has never happened, to my knowledge—but correct me if I am wrong. Why is it different for Adani?

Ms Collins: The difference in this case was that the approval for the original Adani project happened around
the same time as the Reef Trust was established. It has been part of conversations with Adani over a couple of
years that this was a possibility for Adani. It is only recently that the variation has been made, but it has been in
conversations with Adani ever since the Reef Trust was set up.

Mr Cahill: You are looking for confidence in how robust the governance arrangements are around the Reef
Trust.

Senator WATERS: Well, I will not get that, but I am looking for details.

Mr Cahill: What we will do is take that on notice. It is important to realise that the Reef Trust is not only
informed by an Independent Expert Panel that is chaired by Professor Ian Chubb but is also overseen by the Reef
Trust Joint Steering Committee, which has senior officials from both the Commonwealth and the Queensland
government. That is the headline. As an Australian government, our advice is to make sure that you would not put
something into something that was not robust. As a delegate under the EPBC Act, you have to be confident that
the offset is being delivered and the mechanism is enabled to do that. As a delegate you turn your mind to two
things—that what you are trying to offset is being offset and that the vehicle you are choosing to do that has the
appropriate governance around it to give you confidence that it will be delivered. We will give you advice on our
confidence around the Reef Trust as a vehicle to be able to deliver that offset.

Senator WATERS: What I would like you to advice me specifically is whether or not the minister's
involvement is diminished by no longer having this as a marine offset strategy under the permit conditions but
instead using the Reef Trust pathway, whether the community has the opportunity to even see what was the
strategy and is now just some amorphous blob in the Reef Trust and the level of diminution of community
involvement. I expect to be disappointed on both fronts, but I would await your response.

Mr Cahill: I am hopeful you will not be disappointed. We will take that on notice.

Senator WATERS: I want to move to the issue of timing. Under the previous nine conditions that have been
revoked for Adani to be consolidated into this new process, the previous conditions required that the $450,000 be
paid annually, starting within a month of approval. Now, they do not have to pay until construction commences.
The company has gone on record saying that they will not start construction for quite a while yet, because it is at
the end of pipe and they have to do stuff back at the mine first. So they get a delay in when they have to cough-up
their $450,000. This is all very pertinent to fact that they are not in such great financial straits. Why is this
government allowing them to delay payment when previously the conditions said that they had to pay within a
month of the approval having been issued?

Mr Knudson: I will turn to Ms Collins in a second. It is very, very standard practice that what you want are
the offsets to be established et cetera at the time of the impact happening on a matter of national environmental
significance. That is why it is completely reasonable that you would do that at the time of commencement of
construction as opposed to in advance.

Senator WATERS: Why did the original conditions say within one month of approval?

Mr Knudson: I was giving you the general principles behind how we do our offsets. In terms of the specifics
on that particular condition, that would have been a decision for the minister at that time.

Senator WATERS: The minister under Minister Hunt required the money within a month, but now under
Minister Frydenberg he is happy to wait for the money to come on the never-never.

Mr Knudson: The point is that you want the offsets to be in place at the time of the impact. That will not
happen until construction begins, at the earliest.

Ms Collins: The payments are required to be made annually until the year 2053. If Adani was to commence
the activity this year then the first payment would be made this year?

Senator WATERS: They said they will not, though.

Ms Collins: But the payments are required to be made continually each year until 2053.

Senator WATERS: Or until they go totally broke. Who asked for this change of conditions? Did Adani ask
for this or did the minister ask for this?

Ms Collins: As I said earlier, Adani did not specifically ask for it. It is a position that the department took up
and it is in line with the department's commitment to the Reef Trust, and it is in line with the view of getting the
broader strategic outcomes at a larger scale, consistent with the Reef 2050 Plan, and it takes away from this siteby-site
decisions made by individual companies. So the objective is to the broader environmental outcome that is
consistent with the Reef 2050 Plan.

Senator WATERS: Did the minister have to tick-off on that approach?

Ms Collins: The minister's delegate did that.

Senator WATERS: Are you telling me that the department of their own volition decided to, in my view,
weaken—in your view, you say they are not—the conditions on Adani's approval, with no involvement from the
minister and no involvement from Adani?

Ms Collins: The department has delegations to make decisions to vary conditions of approval. In this case, it
was an ongoing conversation with Adani over a period of a couple of years and it was consistent with the
department's policy position and the establishment of the Reef Trust.

Senator WATERS: So you are now saying there was ongoing conversation between departmental officials,
who ultimately made this decision, and the company over several years and ultimately Adani have gotten their
way and have had their conditions changed, and the pay cheque is not due for years.

Mr Knudson: Senator, the characterisation that Adani has got their way—

Senator WATERS: Did they ask for this or not?

Mr Knudson: We have been very clear. We believe that the Reef Trust is a more integrated and strategic way
to invest in the reef, so we have, as Ms Collins has talked about, a number of projects that have already made
contributions to the reef. What we did at that time was say, 'It can either go like a traditional offset or it can go
into the Reef Trust.' The Reef Trust did not exist at that time but we enabled that possibility. This is catching up
with that type of process. This is going to take investments from a whole range of players beyond government to 
achieve the types of outcomes that we need on the reef. This is about doing that and making sure that those
decisions are made in an integrated and strategic way. It is not Adani getting its way; it is the department working
with proponents to be part of the solution in trying to bring together funds towards—

Senator WATERS: Did Adani object?

Ms Collins: No, Adani agreed.

Senator WATERS: They did. What a surprise!

CHAIR: Senator Waters, I do not think this extra editorialising is actually very helpful. Perhaps you could—
Mr Cahill: Senator, it is not an unusual practice conditions based on an ongoing engagement with a company
that holds an approval. Our focus is on what it means for the outcome. To give you an idea, as of 2 May this year
we have had 101 approved projects that have had conditions varied this year alone. So this is not an unusual
practice.

Senator WATERS: After ongoing discussions with the companies involved?

Mr Cahill: Ongoing conversations with companies. Sometimes companies approach us and sometimes it is
just in a normal engagement. What I would say with Adani is that in the conversation with them—and I was not
involved in that—they would have wanted to understand how the Reef Trust would work before they agreed to
any variation, which is what you would expect of any company when they are trying to ensure that they accord
with the conditions we have set.

Senator WATERS: Was the minister briefed on this proposed change?

Ms Collins: No, he was not.

Senator WATERS: Are you telling me that in this highly charged project, which has been in the news for the
last three months solid, the department was proposing to change the conditions and did not think it necessary to
brief the minister?

Ms Collins: The variation was consistent with government policy around the establishment of the Reef Trust.
The Reef Trust was set up specifically to enable offsets to be delivered from proponents who had approval under
the EPBC Act. The decision was made in accordance with current government policy.

Senator WATERS: Has the minister subsequently been briefed?

Ms Collins: The minister's office has received information. I do not know whether the minister has been
briefed personally.

Senator WATERS: Did they ask for it or did you provide it?

Ms Collins: I would have to take that on notice. I cannot recall.

Senator WATERS: Thank you.

Mr Cahill: Senator, I will flag that we regularly update the secretary of the minister's office on the vast
majority of our administrative decisions. We will check what we briefed the minister's office on.

Senator WATERS: I have two further questions before I move on to other issues. My understanding is that
under the Reef Trust architecture the government is free to use the $450 million, if and when you receive it, on
whatever projects you like, as opposed to the earlier marine offset strategy, which was specified in the condition
that required it to be spent on particular matters—turtles and the like.

Mr Knudson: What I said in my opening comment was that this is the same amount of money over the same
period of time to protect the same matters that are impacted. That is what we will undertake, to make sure it
occurs for any funds spent by the Reef Trust associated with this project.

Ms Collins: In fact that was specifically noted in the variation of conditions.

Senator WATERS: Under the previous conditions, my understanding was that the rule about not having more
than 10 per cent of your offset money being spent on research applied, whereas now that it is in the Reef Trust
pathway it will not have limitations. Is that correct?

Ms Collins: No, the note says:
The contribution to the Reef Trust will be used to support conservation activities that are designed to offset the residual
impact of the proposed action …

It is exactly the same as Mr Knudson has pointed out. The money will be spent on exactly the same offset as it
would have been required to do under the previous conditions.

Mr Knudson: The Reef Trust does not actually get into a lot of funding of research et cetera. It is heavily and,
I am pretty sure, almost exclusively focused on on-the-ground actions, so it is probably going to be higher than 90
per cent. Again, that is speculation on my behalf.

Senator WATERS: Let's not speculate until we have proper details. But the fact remains they have had their
conditions changed, you say, at the behest of the department with no objection from the company. That means
they do not have to cough up $450,000 until a point in the future so they have had a financial reprieve, which of
course they are stoked about. Again, can you please clarify for me what involvement the minister had in this?

CHAIR: I think that has been asked and asked. You have just about used your 20 minutes and you did say you
had other things. You can keep on asking the same question. That is fine. But that has been answered at least three
times.

Senator WATERS: If there is anything you have got to add, I would be happy to receive that.

CHAIR: Are you moving on with this particular topic?

Senator WATERS: Not yet if that is okay. Sorry, Chair, I might need a little more time.

CHAIR: If you wrap up on this point, I will go to some of your other colleagues. Initially when we discussed
this, you had 20 minutes.

Senator WATERS: It has gone very quickly.

CHAIR: There was a lot of repetition there, Senator Waters. How you use your time is up to you.
Senator WATERS: I would love another 10 minutes if I could.

CHAIR: It is up to your colleagues.

Senator WHISH-WILSON: You can have my time.

Senator WATERS: Thank you, Pete. Hopefully we can all have our time given that we are four hours early.

CHAIR: We already have an agreement to work with the officials through that dinner break so you will have
another 13 minutes to be precise.

Senator WATERS: I ask now about the requirement for Adani to publish management plans on their website
within one month of them being approved. I asked you this and you responded in question on notice No. 77
saying that the biodiversity offset strategy was approved on 7 October 2016. However, it was not published on the
company's website as it was meant to be until approximately March. Can you tell me what date it did go online
and why the delay.

Ms Collins: The requirement to publish documents online did not specify strategies and this was an offset
strategy. However, given that it is normally routine practice for offset strategies to be published, the department
talked to Adani about the benefits of having the strategies published. It was subsequent to conversations we had
with Adani that they agreed to publish their offset strategies.

Senator WATERS: So you had to ask them to do the thing that would normally be done?

Ms Collins: It was not specifically required. In the approval it referred to—from recollection—management
plans and not strategies so, by the strict definition, it was not specifically required. However, as I said, from a
departmental perspective, it would be routine practice to publish an offset strategy so that is why we engaged with
Adani and they subsequently did publish their strategies.

Senator WATERS: How long did it take them to publish after you had to explain the basic obligations to
them?

Ms Collins: It was only within a matter of a couple of weeks.

Senator WATERS: So what was the overall delay? It was meant to be within a month. It was approved in
October. What was the date that it went online? What was the overall delay?

Ms Collins: It was published online in mid-April.

Senator WATERS: You said it took them a couple of weeks after you asked them to actually do it. Did they
explain the reason for that delay?

Ms Collins: Yes they did.

Senator WATERS: What was their explanation?

Ms Collins: Originally they were reluctant to publish the strategies because they contained information that
they considered commercial-in-confidence around the specific properties they were looking to acquire. The delay 
was caused by them wanting to go through and redact any of that commercial-in-confidence information. Once
they had done that, they published it.

Senator WATERS: In a redacted form?

Ms Collins: Yes, that is right.

Senator WATERS: Is there any obligation for them to publish the full strategy or are they entitled to redact?
Ms Collins: As I said, the technicality in the definitions on the on this one was that they did not have to
necessarily publish a strategy. However, as I said, it is normal routine departmental practice to require offset
management plans to be published and that is why we took it upon themselves ourselves to talk to Adani about
this particular case.

Senator WATERS: Can you clarify for me if they are also obliged to create a biodiversity offsets
management plan or if their conditions only require them to create a strategy?

Ms Collins: There are a number of management plans required.

Mr Cahill: I understand that one of their requirements is a biodiversity offset strategy, which they must
deliver to us. We have a draft with us at this stage for consideration.

Senator WATERS: You have a draft of the strategy?

Mr Cahill: Of the biodiversity offset strategy.

Senator WATERS: Is that the same one that we have been talking about, which was published in April, or is
that a different strategy again?

Ms Collins: The biodiversity offset strategy, as we say, was approved in October 2016. They are also, as you
say, required to prepare an offset area management plan. That was submitted to the department on 7 February this
year and the department has provided some initial feedback to Adani on that plan.

Senator WATERS: They submitted a draft, did they?

Ms Collins: That is right, yes. The department is awaiting feedback from Adani on that one.

Senator WATERS: So you have sent it back to them with feedback and you are waiting to hear from them?

Ms Collins: That is right.

Senator WATERS: Do they have a time frame on when they need to have it back to you by?

Ms Collins: Only that approval is required before commencement.

Senator WATERS: They will then, presumably, be required to publish that within a month of it being
approved, if it is approved.

Ms Collins: Yes.

Senator WATERS: Okay. Thank you. Can I move very quickly to a different project, the INPEX Ichthys
project—the North West Shelf offshore gas. In a few sentences, because I am conscious that I have had a lot of
time, can you give me an update on their work to deliver the western Top End marine megafauna program. I
understand they are many years overdue and very late with payment.

Ms Collins: I would have to take that on notice.

Senator WATERS: Is there anyone that is here who has that detail?

Mr Cahill: We will have to take that one on notice. We do not have the detail with us.

Senator WATERS: I will run through these just in case you can shed some light on them. I am interested in
what the cause of the delay is and who the company have been consulting with in order to develop that megafauna
program. What evidence is there that they are actually doing anything to create this program? Can any of those be
answered yet?

Mr Knudson: No, but it is helpful to have the list of what you want us to answer so we can make sure we give
you a comprehensive response to the questions on notice.

Senator WATERS: I understand that the funds under that program were meant to flow in the third quarter of
2016 and that has not occurred yet. I am interested in why and what the hold-up is. Is it coming or are they trying
to amend their conditions in the manner, perhaps, that Adani has benefited from, a deferral of payment falling
due. On a related issue, I am interested in whether they have identified a final site for the marine reserve. Again,
still can't help?

Mr Cahill: We will get the full list from you, Senator.

Senator WATERS: Under their Coastal Offset Strategy they were meant to have announced those final sites
in January 2014, and that was obviously three years ago. So why has the department let the company default on
their obligations for more than three years? I am keen to get a response on that when you can.

My final line of questioning is about Toondah Harbour, which is obviously in Queensland, where they wanted
to reclaim 20 hectares of Ramsar wetlands for all sorts of residential and commercial development. I asked about
this and you have given me a QON that says, 'Advice is generally provided from the wetlands areas of the
department.' Was that the case in relation to the Toondah Harbour referral the first time they sought approval?
Obviously they have now withdrawn it. Was the wetlands advice provided that first time?

Mr Barker: With regard to the question you asked in relation to the original Toondah Harbour referral: in
November 2015 there was advice sought, as is the department's ordinary practice, from a number of areas within
the department that included the department's wetlands area.

Senator WATERS: Sure. Was it provided though? That is what I am interested in.

Mr Barker: It was provided to the assessment area of the department, yes.

Senator WATERS: What was the general nature of that advice?

Mr Barker: The general nature of the advice was about impacts on the ecological components of the wetland,
including migratory birds. The advice went through the impacts and gave an indication of the likely scale of those
impacts against the relevant ecological components of the wetland.

Senator WATERS: Did it have a recommendation as to whether the referral should proceed, based on the
scope and scale of those impacts?

Mr Barker: The advice we get from line areas is generally around the more factual matrix, if you like, of the
scale of the impacts because the decision about whether there is likely to be a significant impact is then a decision
for the decision maker, either the delegate or the minister.

Senator WATERS: Did that advice come to a suggestion about whether the extent of the impacts was likely
to be significant?

Mr Barker: From recollection, no, it was not. But I am going on recollection and so I would need to take that
on notice.

Senator WATERS: If you could check on that, that would be very helpful. Are you able to provide a copy of
that advice?

Mr Barker: Yes, I can. A copy of that advice has been provided under FOI and so I can provide you with a
copy of the advice as it was provided under FOI.

Senator WATERS: I was not aware; I thought it had been heavily redacted with the FOI provisions. I am
after the unredacted version.

Mr Barker: There has been a number of documents provided in response to the FOI request. There have been
redactions because the project is at the deliberative stage and so there has not yet been a decision on the referral of
that substantive project.

Senator WATERS: Since that referral has been withdrawn and they are now putting in a fresh one, surely it is
not deliberative anymore because it is a referral?

Mr Barker: It is connected with the same overall project. There have been changes to the project with the
new referral but it is still the same project.

Senator WATERS: Will you go back and seek fresh advice from the wetlands section?

Mr Barker: Yes, that is right.

Senator WATERS: In which case the first version is surely no longer deliberative, if you are going to go back
and get a second bite?

Mr Barker: The advice is essentially about the same project. The advice we would expect to get on the
second referral would be informed by the issues that were taken into account the first time around, subject to any
differences in the referral that the department has now received.

Senator WATERS: If you can provide as much of the first advice on the first referral as you can, that would
be deeply appreciated. Can I also ask what meetings the department has had with the proponents in relation to the
withdrawal of the original referral?

Mr Barker: There has been a number of discussions with the proponent prior to the re-referral, if you like, of
the new project. I would have to take on notice details of dates and the like. It is very frequent for the department 
to have quite regular engagements with the proponents as they refer projects and then move through the
regulatory steps.

Senator WATERS: If you could take on notice the dates of those meetings or communications about the
withdrawal, that would be great. Can you outline for me the key differences between the old proposal and the new
one?

Mr Barker: In summary, the primary difference with the new proposal is the clarity around some buffer zones
between the high-tide roost site on Cassim Island—

Senator WATERS: Sorry, can you say that slowly?

Mr Barker: There is a variation to the boundary of the project, particularly in relation to a high-tide roost site
at Cassim Island. The variation of the boundary is between, say, 100 and 200 metres of that high-tide roost site at
Cassim Island, which is mapped in the new referral. There has been some other more assessment-focused changes
to the referral around doing an initial referral-level assessment of the likely impacts against the ecological
character of the wetland. The proponent did conclude that, in their own view, there was likely to be a significant
impact.

Senator WATERS: That is what you will get with the reclamation of 20-plus hectares. Are they still
proposing to reclaim the 20-plus hectares?

Mr Barker: Yes, the referral refers to a reclamation of about 40 hectares from memory.

Senator WATERS: It is 40 hectares now?

Mr Barker: My memory is that is in the referral, yes.

Senator WATERS: So they want to reclaim even more than the first time?

Mr Barker: My memory is that the reclamation size is roughly the same.

Senator WATERS: So my figure is wrong there. Can you provide me with as much information as you can
about that second referral and where it is in the process? Can you also provide any advice that has been sought so
far, which you are able to share, particularly on the wetlands impact?

Mr Barker: I need to be clear about this because, obviously, we have had advice on the first and advice on the
second that we are yet to receive. You are asking for the advice from the department's line areas on the first
referral, and we can provide that.

Senator WATERS: I am, but anything you could give me on the second would be helpful as well.
Mr Barker: The second one is currently open for public comment until Thursday, and then a referral decision
on that will be due on 8 June.

Senator WATERS: Thanks very much.

Senator RHIANNON: I understand that recommendations from the Australian National Audit Office when
they undertook the audit into the management and compliance arrangements of the Department of the
Environment and Energy for governing the wildlife trade have all been adopted. I wanted to ask some questions
about that.

Mr Knudson: Unfortunately, that section of the department is 1.4.

Senator RHIANNON: You're not serious!

Mr Knudson: It is Mr Murphy. That being said, if you have specific questions that you want to table on
notice, we can definitely deal with that.

Senator RHIANNON: Seriously; he is not here?

Mr Knudson: That is correct. I am sorry about that.

CHAIR: Is there anyone else who can answer Senator Rhiannon's question?

Dr de Brouwer: Mr Murphy is the expert.

Senator RHIANNON: Okay. Can we do platypuses? I asked some question about platypuses last time. They
said that two platypuses were proposed to go from Taronga Zoo to San Diego Zoo. Can I ask about that please?
Dr de Brouwer: That is wildlife trade.

Senator RHIANNON: So it is not this one either?

Mr Knudson: No. Sorry.

Dr de Brouwer: It is 1.4.

Senator RHIANNON: Okay. I have quite a few questions about how the recommendation was going and
platypuses. If they are 1.4—

CHAIR: I do have some sympathy with Senator Rhiannon because she had been patiently here when we did
go through 1.4. Are you able to put these on notice?

Senator RHIANNON: It looks like I have to if the people are not here.

Senator WATERS: Could we possibly call them tomorrow?

CHAIR: Secretary?

Dr de Brouwer: Yes, we can see if Mr Murphy is available for tomorrow.

CHAIR: Could you check to see if he could be available first up for maybe 10 minutes for Senator Rhiannon?

Dr de Brouwer: Yes, we will check that now.

CHAIR: If you let the secretariat know then we will—

Dr de Brouwer: Okay.

CHAIR: Senator Ludlam.

Senator LUDLAM: I am going to take you right across the country to the north-east Goldfields of WA. I put
a letter to Minister Frydenberg on 23 February about 11 species of subterranean fauna at the proposed Yeelirrie
uranium mine site in WA. It related to processes that were afoot under state environmental approvals. Is there
anybody at the table who we can have a brief conversation with about that matter?

Mr Edwards: I can help you with that.

Senator LUDLAM: That is fantastic. As a high-level overview, can you sketch for us to what degree this
project is currently under approval by the federal minister or the department?

Mr Edwards: Sure. As you mentioned, it has been subject to a state assessment process. Under
Commonwealth law we deemed it to be a controlled action on 19 June 2009, so it dates back some ways.

Senator LUDLAM: It does.

Mr Edwards: There was an agreement that that assessment approach would occur through an accredited
assessment with Western Australia.

Senator LUDLAM: Unlike my colleague Senator Waters, I am not a specialist in environmental law. Is that
the same as delegated assessment? Is that when it is handed back to state authorities?

Mr Edwards: Yes. It is very similar to the bilateral arrangement. It just means that on some we have to strike
particular terms on how the assessment will proceed. That is the best way to think about it.

Senator LUDLAM: So was the 2009 trigger as far as the Commonwealth was concerned because it was a
nuclear action, or did you have threatened species matters on the Commonwealth minister's desk as well?

Mr Edwards: We certainly have to consider it as a nuclear action. That throws it into considering all-ofenvironment
impact, and so by default it will pick up any other matters as part of that. Essentially, the process
through the state approval process was that at 16 January this year the state minister granted approval for that
project. That is the point at which the Commonwealth assessment on our side of the business commenced.

Senator LUDLAM: Okay. I am not going to ask you to speak for your Western Australian colleagues, but tell
me what happens with your process at that point, where you pick up the ball.

Mr Edwards: Essentially, we consider the assessment material they have pulled together and we have to
assess that, again, against all the matters that we would consider. They do have some slightly different
considerations that they take at the state level. They certainly cover our actions, but they can also consider matters
that we do not necessarily consider.

Senator LUDLAM: As far as your assessment is concerned, this is a little bit different, isn't it? For example,
when we were in discussions with you a couple of months ago about assessments for the Beeliar wetlands—no
nuclear trigger, thank goodness—the Commonwealth's remit started and finished with a couple of federally listed
species and your minister was quite tightly constrained as to what other matters he could consider. Is that right?

Mr Edwards: That is correct.

Senator LUDLAM: Okay—whereas, if it is nuclear action, your remit is actually much broader. Spell it out
for us, if you could. You are welcome to paraphrase the act if you do not have it in front of you.

Mr Edwards: Yes. Probably the best way to think about it is that normally, as you stated, we would consider
matters of environmental national significance; there might be a threatened species, for example, listed under 
national law. For all environment assessments, we would also turn our mind to how state-protected matters, for
example, have been considered in that process, and other general impacts in that area—in this case, of course,
looking at how the states assess those impacts, and considering it in a much more wholesome manner.

Senator LUDLAM: Okay, thank you. Where are you up to? I know I am not allowed to come in here tonight
and ask when we can expect a decision—but when could we expect a decision! For example, are we weeks away,
months away, years away?

Mr Edwards: I think in the months.

Senator LUDLAM: Senator Ruston is looking at you very keenly.

Mr Cahill: It is very difficult because, as an assessment officer is working through material, it is a question of
how much they need to delve, so it is very difficult to predict. I think it is a matter of months, but how many
months is hard, because that is about what doing a thorough assessment involves.

Senator LUDLAM: Yes, indeed.

CHAIR: I think you answered your own question at the start.

Senator LUDLAM: What is that?

CHAIR: You answered your own question, I think, at the start.

Senator LUDLAM: No, no. I am genuinely interested. This is a little bit different to any that I have been
involved in before. Okay. But some matters you do delegate back to state authorities if you think the proper
expertise is there?

Mr Edwards: Sure.

Senator LUDLAM: Fine. I am presuming, then, you are aware—this is incredibly rare—that the state EPA
actually refused to grant approval to the project, which was then their advice to the minister. He can take a
different view if he wants. But I am presuming you are aware that the EPA actually knocked this one back, on the
grounds that they thought that there was a risk of extension to some of those subterranean fauna species.
Mr Edwards: That is correct. I am aware of that.

Senator LUDLAM: You are? Okay. Then the minister says, 'Too bad, don't care; this is going ahead anyway.'
Can you tell us how either your act or your judgement is affected when your state colleagues say, 'There's a real
risk of extinction here; this should not proceed,' and the minister says, 'That's too bad.' Does that impact on your
judgement or your assessment at all?

Mr Edwards: I suppose that what we need to unpack in that circumstance are the matters as they relate to
what we control. My understanding is that in that circumstance there was a recommendation, as you said, by the
WA EPA, and that the state minister approved it on a broader consideration of social and economic grounds. We
would have to consider the degree to which we are able to give weight to those types of matters, but our primary
focus are under protected matters under our law. We talked before about the fact that sometimes there are matters
that at a state level can be considered differently than how we would consider them.

Senator LUDLAM: The state EPA are not concerned with social or economic stuff, or whatever other
counterarguments the mining industry might have thrown at them. The state EPA are actually quite tightly
constrained. In their view—which should concern you as environmental specialists and scientists—this would
likely lead to extinctions of some of those species and subspecies. We have established already that you do have
quite a broad remit; you are able to consider these matters. What impact does it have on your decision-making
process when the experts at the state level recommend against?

Mr Edwards: It is pretty hard to speculate in the middle of—

Senator LUDLAM: It is not speculation. This is life.

Mr Edwards: an assessment process.

Senator LUDLAM: No, but this has happened. I am not asking you a hypothetical.

Mr Edwards: Perhaps I can explain the process that we are undertaking now. The process that we have, now
that the assessment is with us, is that we are engaging with the company to better understand what they are
proposing and to better understand whether there are any variations to their proposal that could be made. That is
our starting point, if we go back to first principles. We do need to talk to the proponent in those circumstances and
make sure we clearly understand what is being proposed—

Senator LUDLAM: Have you talked to the state EPA about why they reached that conclusion?

Mr Cahill: To your point about the material in the recommendation that was put forward by the WA EPA—
we look at everything else and we look at all the information that is put in front of us and needs to go the delegate,
to be able to make an assessment report that goes to a recommendation to the delegate. Obviously, if you see a
report from an EPA that says there are matters of concern, we turn our mind to that.

Senator LUDLAM: Are they recommended against? That has happened about three times in my twenty years
circulating around these issues. It is a pretty rare event. Mr Edwards, you said you had spoken to the proponent.
Mr Cahill, you said you had read the documents of the EPA.

Mr Cahill: No, I have not read them. I just said I am aware of them.

Senator LUDLAM: Did anyone involved in this assessment take the time to speak to your colleagues in the
EPA or any of the specialists that they relied on?

Mr Edwards: Absolutely. We talk to the EPA on most projects. Again, as Mr Cahill suggested, we would
consider their findings, and I would expect that parts of our ongoing discussions on that project have been about
making sure we understanding their findings and any concerns they raised.

Senator LUDLAM: What I did in my correspondence, which I referred to at the outset, was propose that the
species of subterranean fauna that the state EPA found were subject to risk of extinction be referred to the
Threatened Species Scientific Committee for assessment under section 178 of your act before any decision is
made. What can you tell us about that suggestion? The minister came back and said you had recommended
against it. I do not have the letter in front of me. Why would you do that—or are you being verballed?

Mr Knudson: The question of whether a species is referred for assessment under the threatened species
committee is under outcome 1.4.

Senator LUDLAM: Do not do that to us. It is late; not as late as it would have been.

Mr Knudson: Unfortunately, the experts are not in the room to be able to answer that. Again, I am very happy
to take that on notice and come back to you on the considerations around this issue.

Senator LUDLAM: Is it just me or is this system broken? Your state colleagues have said that there is a risk
of extinction here. The federal act does have views about ministers not allowing species to go to the wall. Why
was this not a deal-breaker as far as your assessment was concerned?

Mr Cahill: I disagree that the system is broken. The system says that when we have information put in front
of us we need to look at it and do a thorough assessment, and that is what we are doing at the moment.

Senator LUDLAM: Why was that elevated risk of extinction that the EPA found was steep enough to
recommend again the project—very rare—not a show stopper for your assessment?

Mr Cahill: We are still looking at the material that was put in front of us, and that is part of making sure that,
as a robust regulator, we are doing the right thing.

Senator LUDLAM: It is difficult to have confidence in the process. I respect that you have sent the outcome
1.4 folk home but, if they are going to make an appearance tomorrow, I might put some of these questions to them
as well.

Senator WHISH-WILSON: I have a couple of follow-up questions from my questions on notice around
Macquarie Harbour and the department's compliance audit. In the questions on notice, the department said that
they conducted a compliance monitoring inspection, but it was following allegations of noncompliance or when
noncompliance had been received. I have the questions on notice and the answers here, but unfortunately I do not
have a number.

Ms Collins: I can clarify. The department did what was a routine inspection of the marine salmon farming
operation in Macquarie Harbour. After that we received a further allegation of potential noncompliance with the
EPBC Act decision.

Senator LUDLAM: I see. Did you do the routine inspection after a standard period of time? Did you do it
randomly or did something else prompt you to do the routine inspection?

Ms Collins: We do have a program of monitoring and assurance whereby we try and get to most projects that
have approval under the EPBC Act. While we were in Tasmania, we conducted inspections of the three marine
farming operations. We also conducted routine inspections of other activities while we were there. When we are
in a location, we try and maximise the benefit of our visit. It is a program of routine visits to EPBC Act approved
activities.

Senator LUDLAM: Minister, I know this is an issue you take seriously. Did you instruct the department to go
down to Tassie and see what was going on in relation to Macquarie Harbour and allegations?

Senator Ruston: Under the EPBC Act, the minister for the environment has the capacity to instruct in that
regard, so the answer to that is: I do not know what Minister Frydenberg did.

Senator WHISH-WILSON: You did not have conversations with Minister Frydenberg about this?

Senator Ruston: I have had conversations with Minister Frydenberg in relation to the general issue that we
are discussing, but, no, I did not instruct or request that specific action be taken.

Senator WHISH-WILSON: You went to Tasmania in February, correct?

Ms Collins: That is true.

Senator WHISH-WILSON: Following that, there was an allegation of non-compliance. Can you tell us who
that allegation was from?

Ms Collins: I would have to take that on notice. Normally, we do not disclose information like that.

Senator WHISH-WILSON: Following that allegation, you then did additional compliance work. Is that
correct?

Ms Collins: Yes. As part of our ongoing inspection, we then requested information both from the Tasmanian
government and the three marine farming operators in the harbour. We have since received that information, and
we are still reviewing that information.

Senator WHISH-WILSON: I am pushing my luck here, but, in terms of that allegation, could you tell us
whether it was an environment group, or a government body, or a corporation?

Ms Collins: As I said, we do not normally disclose that type of information.

Senator WHISH-WILSON: In relation to the court action by Huon Aquaculture, could you give us an update
on where that is at and what public departments are doing to plan for that, and when you expect hearings to be
completed?

Ms Collins: We are unable to discuss matters that are before the court.

Senator WHISH-WILSON: You cannot even tell us when you are expecting it to go to court?

Ms Collins: All of the respondents must file their individual defences by 1 June.

Senator WHISH-WILSON: The department says, in its responses to me, that you do not normally publicly
release details of compliance activities or of enforcement processes. Is that a blanket policy right across the board
for any of your compliance activities or enforcement processes?

Ms Collins: It is consistent with normal compliance activity, but I can clarify that we are able to make a
statement about the findings once our findings have been concluded.

Senator WHISH-WILSON: Great.

Ms Collins: We just do not normally release things about the process of a compliance activity.

Senator WHISH-WILSON: You said that you could not comment on when it could be concluded. Can you
tell us whether it has been concluded now?

Ms Collins: It has not yet been concluded. There is still new information emerging that we are continuing to
consider.

Senator WHISH-WILSON: I am not a lawyer, but I understand that, if this goes to court, production of
documents may require you to release information such as the details of your processes. Is that correct?
Ms Collins: The standard court processes would apply.

Senator WHISH-WILSON: The EPA, as part of the Tasmanian government, recently announced a new
biomass cap for Macquarie Harbour. Did you have any input into that new biomass cap, or any discussions with
the EPA about that?

Ms Collins: I understand that the Tasmanian EPA has written to the three marine farming operators in the
harbour proposing a new biomass cap. They are still consulting on that, and I believe they are due to make a
decision by the end of the month.

Senator WHISH-WILSON: The reason I am asking is I want to know if you have had any input into that at
all, as to whether your compliance monitoring or your following up of the allegations is going to be incorporated
into the decision of the Tasmanian government. I would hope that they would respect your independent audit of
the situation down there.

Ms Collins: The department's role in relation to marine farming activity in the harbour is as far as it could
have a significant impact on any matters protected under the Act.

Senator WHISH-WILSON: Correct.

Ms Collins: In this case, this is the Maugean skate and the Tasmanian Wilderness World Heritage Area. The
approval that is issued under the EPBC Act is that it was not a controlled action as long as it was undertaken in a
particular manner. Our interest in ensuring compliance is that those particular manners have been implemented.
As you say, that does involve monitoring activity. I am aware that the Tasmanian government has been taking
into consideration the monitoring activity that is being conducted by the companies.

Senator WHISH-WILSON: Specifically, how many wardens, rangers or inspectors have been employed and
given monitoring powers under the EPBC Act under the department's guidance? Have there been any? What
kinds of resources are you putting into the monitoring?

Mr Cahill: We have legal delegations and certification that we issue to compliance officers. We can take on
notice the number of staff that have been involved particularly with this compliance activity.

Senator WHISH-WILSON: Okay, if you could, take that on notice—and whether the project has been
subject to any compliance audit, even a desktop review, previous to that or any other resources that you have—

CHAIR: Senator Whish-Wilson, can I just get an idea from you of how many more questions you have?
Senator WHISH-WILSON: Two more very short questions. I think we are all out of—

CHAIR: That is fine, thank you.

Mr Cahill: We will take on notice the processes, with the qualification that we have not finalised the
investigation yet.

Senator WHISH-WILSON: This is a pretty broad question, Mr Cahill, or for anyone else who might want to
answer it. In your understanding of the history of the EPBC, has a minister's decision to grant approval for a notcontrolled
action in a particular manner—using your terminology from your answers to questions on notice—on a
project ever resulted in an impact that has led to either an endangered species becoming critically endangered or
the extinction of an endangered species?

Mr Knudson: I am pretty confident in saying, no, that has never happened, but we would want to take that on
notice and come back definitively on that. But I am pretty confident.

Senator WHISH-WILSON: I hate to ask the question, but are you pretty confident that that is not going to
happen with the skate in Macquarie Harbour? It seems to be heading that way.

Mr Cahill: The particular matters were very much focused on monitoring and then there being a response
where there were impacts on the levels like benthic levels. We are watching that quite closely to see if the actions
being taken down in that harbour are consistent with that and ensuring that the matters we have protected are
protected.

Senator WHISH-WILSON: This is my last question. In relation to Tassal's expansion of salmon farming at
Okehampton Bay, in the south-east of Tasmania, around Orford, I just want to clarify: have you been approached,
or has Tassal approached the department, in relation to this project? At what point does a proponent need to
approach the department to get approval around what is an action and what is not a controlled action under
EPBC? By 'proponent' I mean presumably Tassal, but perhaps it could be the state government.

Mr Cahill: I am aware that Tassal has actually approached us, but I will have to get an update, unless Ms
Collins has the status of that approach about Okehampton Bay.

Ms Collins: I can recall that we have been approached, but I just cannot recall exactly well enough to say
confidently the decision on that one, so we will have to take it on notice.

Senator WHISH-WILSON: You potentially have already made a decision on that then?

Ms Collins: No, in terms of whether the matter needed to be referred or not.

Senator WHISH-WILSON: Okay, great. I was particularly interested in the threatened ecological
community down there, the giant kelp forests, as well, because that area has always been on the map of where
they are expected to be, but we know there are not many of them left. I will put some questions on notice to you
about that. Thank you.

CHAIR: Thank you very much. Secretary, was there an additional matter?

Dr de Brouwer: Yes, Chair, just for tomorrow morning: in terms of Senator Rhiannon's and Senator Ludlam's
questions, Mr Murphy and Mr Richardson will be here at nine to address those questions on 1.1—on 1.4.

CHAIR: That is greatly appreciated, thank you. Given the confusion between the two programs, I am grateful
for your ability, so we will make sure that we adjust the program slightly and let the two senators know. But,
Senator Whish-Wilson, it will only be 10 minutes for the recall; otherwise we will start eating into—

Senator WHISH-WILSON: That is fine.

CHAIR: That formally concludes today's hearing of the environment portfolio. There is a bit of a change for
the program tomorrow. We will start with 1.4 for 10 minutes, and then, as you are hopefully aware now, we are
starting with program 3.1, Antarctic science and policy, and just making a couple of small changes. Thank you
very much, Secretary and Minister and all the officials, for your expeditious answering of questions, and
particularly thank you to my colleagues for your very good will and sticking to time. Thank you very much.

Committee adjourned at 19:35	
'''

if len(re.findall('\$ | \% | percent | per cent', content)) < 2: #With more time, I can properly classify the input doc as either being facts & figures type or discussion type.
	papyrus = function_2(content)
else:
	papyrus = function_1(content)

print title + '\n'
print papyrus

#Result/summary:
#7354 lines of text cut down to 816 lines, extracts key facts and figures, and picks up on corrected statements.

'''
ENVIRONMENT AND COMMUNICATIONS LEGISLATION COMMITTEE Estimates (Public) MONDAY, 22 MAY 2017

CHAIR (Senator Reynolds): Welcome. The Senate has referred to the committee the particulars of proposed
expenditure for 2017-18 for the Environment and Energy and the Communications and the Arts portfolios, and
certain other documents. The committee may also examine the annual reports of the departments and agencies
appearing before it. The committee is due to report to the Senate on Tuesday, 20 June 2017 and it has fixed
Friday, 7 July 2017 as the date for the return of answers to questions taken on notice. The committees proceedings
today will begin with general questions of the Department of the Environment and Energy and will then follow
the order as set out in the program.

Dr de Brouwer: Yes, please, Chair. I have some very short comments. On outcome structure, usually we give
a mud map. There have been some changes to the outcome and program structures of the department since the
2016-17 portfolio additional estimates statements. The number of outcomes for the department has been
simplified from five to four. They are broadly environment, climate change, energy and the Antarctic. The major
changes are that land sector initiatives, which were largely the Biodiversity Fund, were previously a separate
program, 1.3, and in sustainable management. They are now included in program 1.1. Program 1.2, environmental
information and research, now includes water matters related to coal seam gas and large coalmining development,
which were previously part of a water program in 4.1. Other water functions have been moved to a new program
1.3 within the environment outcome. To assist you, I have tabled a guide that links the new outcome structure to
key programs and issues. We have tried to fill that out as much as possible.

We have tabled responses to all the 333 questions on notice from the February additional estimates hearing. Of
these questions, around three-quarters were written questions after the hearing. With two estimates hearings left
this year we are currently on track to have around 1,000 questions this calendar yearÔÇöwell above the 10-year
average of about 600 questions a year. As well as providing responses to questions on notice, we have
participated in and responded to an increasing number of parliamentary inquiries. Last year, in calendar year
2016, we had a significant role in 14 inquiriesÔÇö11 Senate, two House and one joint inquiry. In the first four
months of 2017ÔÇöthis calendar yearÔÇöthe parliament has initiated a further 10 inquiriesÔÇösix Senate, two House
and two joint inquiries. So that is 24 inquiries. I note that this is resource intensive. We take our obligations to
parliamentary scrutiny very seriously and as such we will endeavour to answer as many questions as possible over
the next two days and we will endeavour to answer all questions on notice from this hearing in the timeframes
requested.

Mr Sullivan: From memory it has been twice so far. If that is incorrect I will correct it over the period of
estimates.

Mr Thompson: It is significantly different in terms of the platform. What is pleasingly not different is that we
are using a similar format to the 2011 report, so you are able for the first time in the history of State of the
Environment reporting from when the EPBC Act came in to track more directlyÔÇönot completely, but more
directlyÔÇöfrom the 2011 report to the 2016 report. We have had very positive feedback about that. We have also
had very positive feedback about the digital platform and the way in which people are able to work with it. There
are over 300 maps and charts in that report. That material can also be taken and pulled apart and assembled in
ways people want to use it to answer particular questions that they have, whether they are policy questions or land
use or natural resource management questions. All of the data sources that underpin the findings of the State of
the Environment report are accessible in the layer beneath the digital platform. In terms of the transparency of the
reporting and the independence and rigour around the reporting there has been very strong feedback, and we are
building on a very good product.

Senator MOORE: The recent report had some fairly serious comments about lack of leadership, policy and
follow-up on policy action. It was through some of the statements that were made about the progress between
2011 and the latest one. I know the department cannot comment on that. Minister, can the government comment
on some of those findings that were in that report? They are clearly around lack of leadership. The State of the
Environment report questioned the degree of leadership around environmental issues.

Mr Thompson: And a responsibilityÔÇönot only for those governments but for other partners, including the
business community, NGOs and the wider community. In that context they did point to the need for the various
partners to collect around some of the clear priorities identified in State of the Environment 2016 and the
challenges there and to pursue action together to meet those. Coming back briefly to the latter part of your earlier
question about the department's response to those thingsÔÇöas I said, we have a small SoE team ongoing, but the
State of the Environment is used across the department to help prioritise our spending programs, to help inform
our advice to government on policy, to identify emerging risks in relation to regulation under the Environment
Protection and Biodiversity Conservation Act and, importantly, also to inform the natural resource management
that we undertake ourselves, whether that is in Antarctica or in our parks estate under the Director of National
Parks and the Commonwealth Environmental Water Holder. That information informs us, forms part of our
ongoing work and forms part of the government's response to that report.

Mr Thompson: The SoE did have a chapter on climate change and atmosphere. The other area would be
around Commonwealth marine reserves and the protection provided there, which the State of the Environment
clearly pointed to as an improvement from 2011. The declaration of those reserves and the management plans for
those reserves are currently being considered by the department.

Senator Birmingham: Across all of the different facets of environmental regulation and stewardship there
have been some great strides forward in terms of the collaborative approaches between the Commonwealth and
other stakeholders, particularly the states and territories. Mr Thompson cited some of the work around the
Murray-Darling Basin before. That is one area. I expect that in the next little while we will turn to talking about
the Great Barrier Reef. Despite some of the very serious challenges there, I think the Reef 2050 plan and the work
there of far more integrated collaboration between the Queensland government and the Commonwealth
government that perhaps has historically never been the case in terms of reef management and stewardship is an
important example of us finding ways for the different parts of government to provide a clear national element of
leadership in a coordinated manner. Overall there are several key frameworks that bring together environmental
policy.

Senator ROBERTS: Thank you. Energy is increasing in price. For someone on $40,000 a year income, the
energy prices they are facing at the moment are like driving over a cliff. On our salaries, and on your salary as a
department head, that just becomes a pebble to some people. It is very significant thenÔÇöI would like to know
what the renewable energy target is running at the moment. Is it still 23 per cent?

Dr de Brouwer: It is a megawatt hour target. It is 33,000 megawatt hours a year by 2020. That then becomes
equivalent to, based on the projected demand for 2020, 23┬¢ per cent. We will confirm exactly what that number
is, but it is formally a megawatt hour target.

Senator ROBERTS: What is the basis of the renewables target? On what advice has that target been set?
Ms Evans: There was an extensive review of the renewable energy target done in 2014, and the target was set
through bipartisan agreement at 33,000 gigawatt hours. There was a range of modelling and other work done, and
that report is in the public domain. We can arrange for a copy to be sent to you if you would like.
Senator ROBERTS: On whose advice was that bipartisan decision was madeÔÇöwhat body?

Dr de Brouwer: There was a major review by Mr WarburtonÔÇöthe Warburton reviewÔÇöon the renewable
energy target. I would have to go back to that document, but the government, based on that report and in
negotiations in the parliament, settled on 33,000 gigawatt hours. SorryÔÇöearlier I should have said gigawatt hour,
not megawatt hour.

Mr Thompson: Sure. In the answer I gave to Senator Moore a bit earlier, I indicated that there was not a
general response by the governmentÔÇöa policy responseÔÇöto State of the Environment 2016 but that we now
reflect on the findings of the report and the material there in terms of the policy advice that we give to
government, the spending advice that we give to government et cetera. So, in part, the government's decisions in
recent times around threatened species and providing additional resources for the Threatened Species StrategyÔÇöI
think it is a further $5 million being provided to thatÔÇöreflects some of the findings in the SoE around the
continued pressure on threatened species across the nation and, in particular, the decline in the status of small
mammals in the Australian environment.

Mr Thompson: That is fineÔÇöthank you for clarifying. I am not saying that there will be policy. I do not
expect there will be policies that the government will announce and the government will say, 'This is because of
State of the Environment'. What I expect is that there will be advice to government and continuing to go to
government which reflects the findings of State of the Environment 2016 and which the government will consider
and make decisions on accordingly. In the budget, of course, there was a commitment to a further $1.1 billion for
the National Landcare Programme over seven years. That is obviously a major contribution to a number of the
land management and biodiversity challenges which the SoE identifies.

Senator URQUHART: That goes nicely into my next line of questioning about Landcare. It is $1.5 billion,
did you say, or $1.1 billion?

Senator URQUHART: But can you as a department tell me if that is a lesser amount than in 2013ÔÇöthe $1.1
billion over seven years?

Mr Thompson: I am probably not going to satisfy you with this answer, but it depends a bit on how you
measureÔÇöover what time period you measure it. That seven years obviously includes some additional funding in
the current financial yearÔÇö2016-17. It includes the commitment that was made in the Mid-year Economic and
Fiscal Outlook as part of the deal with the Australian Greens around Indigenous Protected Areas and the National
Landcare Programme. It is the commitment for the rolling cycle of National Landcare Programme funding.

Senator URQUHART: Again, this is to the minister, but if you want to flick it then I am sure you will. Can
you confirm that the allocation of money apportioned to the environment portfolio from the $1.1 billion in the
rollout of the National Landcare Programme moneys referred to in the budget will stay the same as previously?
Will it rise or will it fall between now and 2023?

Senator Birmingham: Just for the record, Senator Urquhart, it is $1.1 billion, not $1.1 million.

To confirm the extent and severity of the 2017 bleaching event, this year authority staff took part in an aerial
survey conducted by James Cook University. The 2017 bleaching footprint was mainly in the central region,
differing from the 2016 event, which was mostly in the northern region. On 28 March, Tropical Cyclone Debbie
crossed the coast at Airlie Beach, causing significant damage in the Whitsundays region of the marine park. The
authority has been working closely with our partners and tourism stakeholders in the region to assist recovery in
that area. This includes providing permissions for active intervention. This is a fairly new thing for us. There is a
short window when, if you turn a coral over, it will survive. It is a small thing. We lifted the normal ban on
touching coral for the couple of weeks that that opportunity was there. A lot of the operators did that and were
grateful for the recognition that there was something that they could be doing. We also returned washed-up
boulders to create structure for settlement of new corals into the deep water. The massive waves were lifting
corals the size of small vehicles up and out of the water. They would be dead, but their structure is important, so
we rolled them back into the ocean.

The protective measures in the marine park play a large role in delivery of the Australian and Queensland
governments' Reef 2050 plan. This is the 35-year management framework supporting the health and resilience of
the Reef.

Dr Reichelt: In a general sense, that is true. It is more or less what we said in 2014 in our outlook report. In
the covering letter to the minister and the parliament in making that submission, I wrote that the condition was
poor and likely to worsen. When that was being written we thought there was more time for the climate impacts
to be dealt with than we have now, or than is apparent from the current intense bleaching. We are also worried
about declines in water quality. There has been a massive effort to improve those, which is ongoing, and will take
a long time as well. Why I agree with Professor Chubb's comments is that it is not just about climate change.

Senator CHISHOLM: Can I clarify, Dr Reichelt, one thing that you said. When you talk about a change of
approach, is that within the current Reef 2050 Plan? Are you talking about change within that current framework?

Senator Birmingham: My understanding is that the communique, which Senator Chisholm has been referring
his questions from, does refer to the Reef 2050 Plan as being a strong foundation. Of course, that is the approach
we have taken in building something with the Queensland government that ensures there is a cross-government
framework and plan upon which integrated action happens through GBRMPA and through other agencies of both
governments and other stakeholders to deal with these challenges.

Mr Oxley: The ministerial forum that was held at the end of last year asked the independent expert panel to
provide it with some advice on what further action could be taken to deal with the impacts of coral bleaching.
That obviously has been given a much greater emphasis as a consequence of the bleaching event continuing into
2017. The independent expert panel met a couple of weeks ago, as Dr Reichelt has outlined, and discussed the
nature of the advice that it would be providing to the two ministersÔÇöMinister Frydenberg; and Minister Miles,
the Queensland Minister for National Parks and the Great Barrier Reef. The panel issued the communique, which
has been discussed already this morning, and it is now in the process of formulating its comprehensive advice to
ministers. That comprehensive advice will be provided to ministers for their consideration at the next meeting of
the Great Barrier Reef Ministerial Forum in mid-July.

Senator Birmingham: Our approach has been to build what is acknowledged to be a strong framework in the
Reef 2050 Plan. We will continue to build upon that, and that includes around 151 different actions within a
strategy that is worked upon collaboratively with the Queensland government. It includes around $2 billion worth
of ongoing investment and action. In addition, there is the $1 billion within the CEFC Reef Funding Program for
particular investment in climate change and water quality projects that can have impact in relation to the Reef
region, key catchments, the reduction in nitrogen run-off, as well as assisting with climate change mitigation
activities. We will continually work to improve these policies and processes, advised and informed by such work.
As you heard, there is a formal review mechanism in place for the Reef Plan. Obviously, these findings will
inform that review and, of course, ministers will have their discussions together about further actions that need to
be undertaken.

Ms Parry: In addition to what Dr Reichelt has outlined in terms of the crown-of-thorns, for which the
Australian government has contributed $18.8 million since 2014-15 to support crown-of-thorns control and
research, we also put in significant investments in the catchment side through the Reef Trust and, previously,
through the Reef program. Most recently, phases 4 and 5 of the Reef Trust were rolled out to address, again,
highly targeted industries and regions. This particular round focused very much on gully and stream bank erosion
across the highest threats and targeting the greatest sources of sediment discharge into the Reef regions.

In addition, we have also leveraged considerable financing through phase 5, with partnerships with Greening
Australia. In total, the Australian government contribution of $7 million was matched by Greening Australia to
bring a total of $14 million into wetland rehabilitation. On the back of that, the Reef Trust also was involved in a
partnership deal with Msf Sugar, looking at lowering and improving behavioural practices and farming practices
for the Msf Sugar farms that feed into that particular mill. The Reef Trust invested $4.5 million and Msf Sugar is
investing $12.8 million. Those are the latest Reef Trust 4 and 5 investments.

Dr Reichelt: The climate change risk is the global one I talked about, and mitigation. That is not part of the
Reef 2050 Plan. It is acknowledged in it as a key issue and a key risk, but it is a separate activity.

Mr Oxley: If I may, you are going to a question about the Reef 2050 Plan, and the department is responsible
for the Reef 2050 Plan. So, if you do not mind, I will give you the initial answer. The Reef 2050 Plan
acknowledges right up-front that climate change is the single biggest pressure and threat facing the Great Barrier
Reef. But the government was also very clear, in partnership with the Queensland government in developing the
Reef 2050 Plan, that its focus was on improving the health and resilience of the Reef so that it was best placed to
deal with the pressures associated with climate change. But the place for climate change policy to be prosecuted
was not through the Reef 2050 Plan but through the government's engagement in the Paris Agreement in the
UNFCCC process. That is very clear from all of the documentation associated with the Reef 2050 Plan. So while
climate change is acknowledged as the single biggest pressure facing the reef, the plan also recognises that the
Reef 2050 Plan is not the place to prosecute climate change policy outcomes.

Senator Birmingham: I think the Reef 2050 Plan makes it clear that action to address climate change needs
to be a priority. It is a priority and we implement that priority through the Paris Agreement and then through the
different policies that we deliver to meet Australia's commitment under the Paris Agreement. But, though the
issues of climate change are very important to the future of the Reef, the Reef is not the only area in which those
issues are important. Climate change takes on a priority in other ways as well as in relation to the Reef. Of course,
as we emphasised before, it requires a global effort to actually sustain serious action in relation to climate change.
That is where the focus lies. It is identified up-front as a priority and then government takes actions to address that
priority in a range of ways. But the Reef 2050 Plan goes specifically to a range of actions and strategies that are
applied and coordinated across the Australian and Queensland governments, in different agencies and
stakeholders, to deal with a range of other factors that impact on the ReefÔÇöbut also areas of mitigation strategies
that directly relate back to what some of the impacts from climate change may be.

Senator Birmingham: Officials may want to talk specifically to the content of the Reef 2050 Plan. They can
probably cite some of the words within it. It identifies that action is important in relation to dealing with the issue
of climate change. It also, in terms of the different plans and strategies and so on that are outlined in the Reef
2050 Plan, includes plans and strategies that relate to the specific areas of adaptation and action that can be taken
in the context of the Reef as it relates to the climate change threat and other threats that the Reef faces.

Dr Reichelt: The focus of the Reef 2050 Plan is all about the resilience of the system in the face of climate
change. That is how it is couched, right up-front. Some of the interlinkages between water quality, crown-ofthorns,
coastal development and even over fishing areasÔÇöwe interpret all of them in light of the resilience of the
system in the face of climate change. We strongly support some of the work mentioned earlier on water quality,
because it is known that corals living in water with elevated sediment nutrient are more prone to bleaching, for
instance. So the act of cleaning up coastal waters is a way of building the resilience of the system. There is four
times the amount of sediment and nutrients coming into the coastal waters than there was pre 1700. I strongly
support the Reef 2050 Plan because it puts the focus on building resilience. That is not meant to detract from or
deny the importance of the national and international policies on climate change, which I have always
acknowledged are extremely important. They are addressing the primary external risk to the Reef. It is just that
the actions that we can take in the marine park are to build the resilience of that system in the face of that. That is
how I see it.

Senator CHISHOLM: You generously let me ask these questions last time. That was your mistake!

Senator WATERS: I will continue on the line of questioning about the Reef 2050 Plan. The independent
expert panel made a clear recommendation that the reef plan be amended to include climate change adaptation
and mitigation actions. Minister, perhaps this one is for you. Is that under genuine consideration by this
government?

Mr Oxley: If I might, rather than the minister, and Ms Parry may wish to expand on my answer. There is a
review of the Reef 2050 Plan scheduled for 2018. We are working on the expectationÔÇöand it is the express
expectation that has similarly been conveyed by the independent expert panelÔÇöthat the sort of issues that they
have raised would be considered in that review of the Reef 2050 Plan.

Senator Birmingham: As you are well aware, in terms of meeting Australia's future commitments in relation
to emissions reductions, policy work on that will follow the expert work that is being undertaken there to see how
we build on the success of the ERF in the future. Of course, that has put Australia in a position where once again
we will meet and exceed current targets. I am confident we will take the right policy responses to meet and
exceed the 2030 targets that Australia has set down as well. Those targets will be met and exceeded, regardless of
other development activities that may occur in other parts of Australia.

Senator WATERS: That is only because we are the biggest emitter. Anyway, moving on. Dr Reichelt, there
has been some research published by the University of Melbourne climate scientist, Dr Andrew King. The
temperatures we have just experienced this year resulted in the second successive event that was, as you said,
unprecedented in the reef's ancient history. He found that temperatures at that level would be 97 per cent likely
every single year by 2050 at the current rate of climate emissions. In your view, would the reef survive that level
of bleaching?

Dr Reichelt: I don't think we know that, really. I have not seen anything precise. I have seen some
projectionsÔÇöand these were made 20 years agoÔÇöof when back-to-back bleaching may occur, which was in the
mid-2030sÔÇöand it has occurred now in 2017. So I think that the general conclusion about reaching these levels
will have a dramatic effect on coral reefs. To be precise about where in the intervening range of temperatures
things would happen, I don't think we have that predictive capability at the moment. That shouldn't stop us from
taking it very seriously and looking to remain under two degreesÔÇö1.5 degrees is what my colleagues tell me is
the level at which a fair proportion of the diversity of coral would survive.

Senator WATERS: In the submission to a Senate inquiry that I instigated with the support of the Senate in
2014, GBRMPA called for a limit of 1.2 degrees. Do you still stand by the 1.2 degrees or have you gone up to 1.5
degrees now?

The reef scientists on the Queensland government expert panel had called for a significant investment in water
quality, to the tune of about $8 billion. Clearly our water quality investments from this government and the
Queensland government is nowhere near that amount. Do you share the view that we need that serious level of
funding in order to genuinely address water quality issues in the reef?

Senator WATERS: Okay. Just on that, I hear you that you say outcomes are important but the outcomes
aren't being met. The target for dissolved nitrogen is a 50 per cent reduction by next year and an 80 per cent
reduction by 2025. At the moment we have only had an 18 per cent reduction. So we are clearly nowhere near
meeting that most important target. Is it your view that the water quality targets under the reef plan will be met?

Dr Reichelt: I have seen those in the independent expert panel. The JCU surveys and the models they have of
this year were in the vicinity of 20 per cent, I think. So that is 30 per cent last year and 20 per cent this year. That
is a modelled outcome, which our preliminary observations say could well be true. We are relying now on the
long-term monitoring program that covers the entire reef, rather than just areas of impact. We will see the 2016
figures released within the next few weeks, I understand. 2017 will take until this time next year. Without wishing
to frustrate the public, it is a very large system and the best data comes from divers in the water at the moment,
rather than aerial surveys. I think the responsibility on us when we do say what we think is happening, we want it
to be credible and able to be relied on. What we are saying is that we agree with the 29 to 30 per cent last year.
That is due for release in a much more detailed way.

Dr Reichelt: We received $2 million a year. But it is not just for the bleaching response. It is for designing the
system that we will need to have in place for the long termÔÇöfor future outlook reports, for the assessing of the
Reef 2050 Plan.

Dr Reichelt: No, it's not. We have 100 park rangers in our cooperative program with Queensland. We have a
$6 million vessel that has been a radical multiplier for their capability to look at this system. The other thing is
that we have managed to harness a cooperative arrangement with James Cook and with AIMS, who also have
vessels. So there is a high level of cooperation and data sharing at the moment. That probably wasn't there 10
years ago. We are better equipped now than even after Cyclone Yasi in 2009. What we need to do is make sure
we are spending our time on resilience buildingÔÇöthe things I spoke about earlier. Monitoring is something that, if
we do it now, we'd be getting ahead of it. These events take a year at least to know what has survived and what
hasn't. Corals that appear recovered can revert. We have to think of it not as a bush fire that we can go out to the
day after and measure what burnt and then count the damage. It is a longer-term process.

Mr Knudson: Back in the 2016-17 Mid-Year Economic and Fiscal Outlook, the government provided an
additional $124 million to GBRMPA over 10 years.

Senator WATERS: Wow! That is so much over 10 years. Compared to the $27 billion to coalmining
companies in cheap fuel, it pales, I'm afraid. But thank you for that clarification.

Mr Knudson: Just before Dr Reichelt talks, I just want to clarify a point that I think Senator Waters made
which is just factually incorrect. It is that the existing Reef 2050 Plan, the first key challenge that it talks about is
on climate change. It talks explicitly about actions under the Paris Agreement as well as adaptation. It does that
for a couple of pages, laying out a number of the items that Senator Waters talked about with respect to the
emissions reduction fund, et cetera. I just want to make sure that that is very clear. It is covered in the existing
plan.

Dr Reichelt: This is a very common question I get. As I am here representing the marine park authority that
manages the marine park, we have made very clear over several cycles of the outlook report where the risks lie,
including maps of impacts that run out to 2050 in the outlook reports. The specifics of the impacts of a mining
operation on the marine park is something we looked at through the lens of our legislation. Commenting on a
specific project is not something that our marine park authority has responsibility for. The specifics of the impacts
of a mining operation on the marine park are something we looked at through the lens of our legislation. To
comment on the specific project is not something that our marine park authority has responsibility for. What we
doÔÇö

Mr Oxley: The trip was managed and conducted by the Department of Foreign Affairs and Trade, with
support from the Great Barrier Reef Marine Park Authority. The department's small contribution was for a couple
of officers from the department, myself included, to join the trip and have the opportunity to see first-hand what
the diplomatic corps saw and to be available to talk about the Reef 2050 Plan should any of the diplomatic corps
want to understand the scope of the government's response.

Mr Oxley: I have not seen those comments, so I do not know whether it is an accurate attribution or not. The
department provided briefing material to the Department of Foreign Affairs and Trade, and it provided a briefing
to the Minister for Foreign Affairs, and, in advance of the trip, both myself and Dr Reichelt had an opportunity to
spend some time with the foreign minister, to brief her on what she could expect to see when she took the
diplomatic corps to the Great Barrier Reef and to bring her up to date with the Reef 2050 Plan.

Ms Parry: I can break down specifically around crown of thorns, but I can give you an overall budget picture
of our various programs and roughly where they are directed towards. In terms of the Reef 2050 implementation,
supporting the implementation of the Reef 2050 Plan our budget from 2016-17 out to 2021-22 is $94.777 million.
The Reef Trust over its lifetime from 2014-15 out to 2021-22 is $210 million. That is the Australian government
contribution and does not take account of any external funding that goes into the Reef Trust. The reef program
from 2014 to 2017-18 is $82.75 million. We have other NHT funding of $6.493 million from 2014-15 to 2017-18.
Similarly, from 2014-15 to 2017-18 there is an additional $27 million from the Biodiversity Fund which is
directed towards the Great Barrier Reef Foundation. We also have the National Environmental Science Program
running from 2014-15 to 2020-21, which is $31.98 million. That makes a total departmental expenditure of $456
million from 2014 to 2020-21. I should also point out that we spend as a portion of that, as I indicated earlier,
$18.8 million crown-of-thorns starfish control.

That is not all of the Australian government expenditure that goes towards protection of the reef. That was
outlined in significant detail in the investment framework that was released in December 2016, which showed that
the Australian government is spending $716 million over five years to support the implementation of the Reef
2050 Plan. That, combined with Queensland government investment, is $1.28 billion over five years to support
the implementation of the Reef 2050 Plan, which is part of the overall government commitment of $2 billion over
10 years. This funding also does not take into account the $1 billion Great Barrier Reef Fund that is administered
through the Clean Energy Finance Corporation, which is debt and equity financing available for reef investments,
nor does it take account of the additional $124 million announced in MYEFO that was referenced earlier.
Senator CHISHOLM: In terms of, for instance, the reef program, which I think was from 2014-15 to 2017-
18ÔÇö

Ms Parry: That program is due to end in 2017-18 and the transition to the Reef Trust picks up investments
from 2017-18 onwards, as well as the implementation of the Reef 2050 Plan.

Senator WHISH-WILSON: On notice, yes. Could you also confirm that in 2016 alone, 531 sharks, including
endangered species such as the great white and the grey nurse, were killed, according to the data from the
Queensland government; and, since the program began, over 50,000 dolphins, dugongs, marine turtles, rays and
whales have also been killed by these lethal technologies.

Senator URQUHART: I know that in 2015 the Abbott-Turnbull government sponsored a review into the
removal of about 127,000 square kilometres from the reserve, which is almost the size of my home state of
Tasmania. At that time, I understand that there was no justification given, but there has been a review. I have not
seen a response. Has there been a formal response to that review, given that it was a couple of years ago?

Ms Barnes: The government's commitment of $56.1 million over four years is in the budget, it is just in
various places in the portfolio budget statement. Some of that money is in the Director of National Parks's
statement, some of it is in the Department of Environment and Energy's statement as administered funds, but the
total is still $56.1 million.

Mr Mundy: There are a range of activities going in reserves that are going on around the Commonwealth
marine reserve network at present, including science activities and particularly in the southeast network, which is
currently under active management and has management plans in place, as well as a series of 14 other reserves
around the country for which management arrangements existed pre-2012. Particularly in those locations, we
have active compliance and active science and management and permitting programs in place.

Senator SIEWERT: As I said, I have your letter. Can we now just confirm what the role of the Northern
Land Council in fact is, given the correction you gave.

Mr Thompson: Senator Moore asked earlier about conferences that the department has participated in in
relation to Sustainable Development Goals. Obviously in addition to those meetings where SDGs have been
discussed in international and domestic settings, there have been two major domestic conferences where the
department has participated. One was the Australian SDGs Summit in Sydney in September 2016 and the other
was the SDG Australia conference in Sydney during November 2016.

Mr Webb: Thanks, Dr Johnson. I also remember that we are at the back end of an injection of forecaster
resources over the last three to four years. There was a review into the Bureau of Meteorology after the 2010-11
floods that resulted in an injection of forecasters to allow us to surge to those extremes. As Dr Johnson says, even
though the big events will stretch any organisation and require a really structured way of responding, we are
certainly at the back end of that project, at the moment, with more forecasters in our forecasting chairs. It is
making the response a lot more ordered. On top of that, as Dr Johnson said, the structured approach to
competencies around the country that allow people to step in wherever they are and standardisation of all of our
services means that we can more easily respond to the ebbs and flows. We are very mindful of the mental health
of all of our employees as well.

Senator URQUHART: I want to talk about the funding allocation. Looking at the Environment and Energy
portfolio budget papers, the Climate Change Authority has been allocated $1.465 million in the financial year
2017-18 and nothing after that. Is that correct?

Senator URQUHART: So that is a significant cut in funding compared to the previous year, which was
$3.529 million. Yet, as I understand it, the staffing level in both of those years is reported as staying the sameÔÇöat
nine. Is that correct?

Ms Thompson: No. I do not think that is quite right. One of the issues to bear in mind is that when the
authority was set up in 2012, it had rather more functions than it has at the moment. So when we were originally
set up, part of our role was to advise on the caps for the carbon pricing mechanism. We also had a role doing the
regular renewable energy target reviews. Both of those functions have been removed as a result of legislative
changes since. So while we do still have some statutory review functions, and while from time to time we are
asked to do special reviews at the request of the minister or possibly the parliament, in fact, a number of our
existing functions are no longer afoot for us. So I think that is partially reflected in the staffing profile. Another
thing to bear in mind is that when we were based in Melbourne there was the need to maintain a dedicated
corporate area just for us. With the move to Canberra in September last year, we no longer need to maintain such
a big standalone corporate function because it is easier to access shared services from the department and other
providers.

Senator URQUHART: Yes, absolutely. Since the CCA was established in 2011, the science of climate
change has progressed significantlyÔÇöis that correct?

Senator Birmingham: This is the government's review of climate change policy. It is to help inform the
meeting of the 2030 targets.

Senator Birmingham: The government has provided $2.1 million in the 2017-18 budget to support the
authority. Funding arrangements for 2018-19 onwards will be subject to the usual budget processes.

Ms Evans: I might make a minor correction. There is a $1.456 million allocation. The briefing that we
provided to the minister was incorrect.

Senator WATERS: Okay. I have $1.465 million and $1.456 million in front of me in the budget papers.
Ms Evans: They are the correct numbers in the budget papers.

Ms Thompson: Our terms of reference are for the work to be informed by independent modelling.
Fortunately, quite a bit had already been done. You may be aware that the authority commissioned Jacobs to do
some modelling for us in 2015. Similarly, the Australian Energy Market Commission also sought modelling from
Frontier, I think, about the same time. So we have those two pieces of work that are already on foot that we can
draw on. In addition, we have sought some further analysis from the Centre for International Economics. They are
working with us to provide some, as I say, further analysis looking at those previous modelling exercises. They
are also doing a bit more work for us that we have asked them to do. So we are hopeful that that will provide a bit
of a fresh eye on that modelling work and help everyone distil the most salient features.

It was a special review research report of August 2016. This report is in favour of an emissions intensity scheme.
That is in the summary at page 11. It is referred to there first up as well as in other parts of the report. Is this one
of the inputs that you will be considering in the context of this report?

Senator XENOPHON: My understanding is that the AEMC are very much in favour of an EIS as per the
December 2016 final report into the integration of energy and emission reduction policy. What happens if there is
a fundamental difference of opinion between the Climate Change Authority and the AEMC? How is that
resolved? What process is there to resolve that? How do you take that into account?

Ms Jonasson: I would also like to clarify that the total amount of funding that has been announced by the
government, the $1.1 billion, consists of the full allocation of funding that is in the current Natural Heritage Trust.
There have been no savings taken out of the Natural Heritage Trust or the allocation of that funding from 2016-17
as well as the $100 million that was announced in MYEFO at the end of last year. So that is where the $1.1 billion
comes from. There have been no savings taken from the Natural Heritage Trust in relation to this announcement
by the government.

Senator URQUHART: Thanks for that. Are you able to provide any more detailed funding by the National
Landcare Program subprogram for 2017-18 and over the forward estimates?

Ms Campbell: We can certainly provide the subprograms under the National Landcare Program for 2017-18.
The programs beyond the forward estimates have not been determined by government, so we are unable to do so
at this time.

Mr Dadswell: The subprograms under the current National Landcare Program include the Environmental
Stewardship Program, which in the 2017-18 year is $11 million; Sustainable Agriculture and Biosecurity
Incursion Management, which is appropriated to, and administered by, the department of agriculture and is $3.7
million; and pests and disease preparedness, which is appropriated to the Treasury and administered by the
Department of Agriculture and Water Resources and is $19.9 million. So that is in addition to the Natural
Heritage Trust that is appropriated to the department.

Mr Andrews: Thank you, Chair. I just got back from Kangaroo Island on Friday, which will be the world's
biggest ever island to have feral cats eradicated. It is one of the priorities in the threatened species strategy. There
are more feral cats than people on Kangaroo Island. The farmers and the conservationists are united to get rid of
the feral cats because of their impact not only on the wildlifeÔÇödozens of species will benefitÔÇöbut also on
agricultural productivity through sheep and chook farming. I think I can give you an update. I can also say that
Australia is very supportive of this issue. Last week, on my Facebook account, I directly engaged on this issue of
feral cats. One hundred and fifty-six thousand people commented or liked it. I worked out that roughly the
equivalent of one out of every 150 people in Australia actually engaged and were interested, and 99 per cent of
them were supportive. To date, to meet the targets of five feral cat free islands by 2020, 10 large predator proof
fenced areas on the mainland by 2020 and two million feral cats culled humanely, effectively and justifiably by
2020, the Australian government has mobilised $30 million. I can table a summary of particular projects. I will
just mention one of them.

Mr Andrews: My work intersects with recovery plans and other documents because my job is to lead the
implementation of the threatened species strategy, work with recovery teams and areas of the department that are
responsible for drafting recovery plans and work with the Threatened Species Scientific Committee. My job
overall is threefold. One is to raise awareness and support for threatened species. I mentioned at the last estimates
that I summarise that as competing with the Kardashians because more people know who they are than what a
numbat, quoll or quokka looks like. The second is to mobilise support from within government. I am proud to say
that we have mobilised $227 million for projects. The funding for those projects is consistent with the science.
Sometimes recovery plans do not have the most up-to-date science. I am reluctant to say that the funding for those
programs is to implement recovery plans because some recovery plans actually do not have the best science, and
the most latest and up-to-date science is what is required.

Mr Andrews: Look, I could summarise by saying that the fundingÔÇöI have provided the tables for this
funding beforeÔÇöcomes from the Natural Heritage Trust suite of programs, being the 20 Million Trees program,
the National Landcare Program and the Green Army, and the National Environmental Science Program. For
example, we have for the first time ever a $30 million hub dedicated solely to threatened species recovery that
comes out of the National Environmental Science Program.

Senator WHISH-WILSON: I have a couple of quick questions on the Tamar River dredging program. Mr
Costello, I understand that the funding runs out in 2016-17, which is the end of financial year 2017. Correct me if
it is calendar year. Could you give me a quick update on the process from here? Are there any feedback loops at
all in terms of the funding that has been provided?

Mr Costello: There was a $3 million commitment over three years. You are correct; it does finish at 30 June
this year. But there was a further commitment made for another $1.5 million for the next three years. So that is
being looked at as a second stage of the work that is being done. That new commitment has been incorporated
into the Launceston city deal that was announced in April, so there is some continuity of the work there.
Senator WHISH-WILSON: So that was announced when the Prime Minister came down to Launceston in
April?

Mr Costello: Yes, certainly. So the major floods in June 2016 had a very large scouring impact. Obviously,
you cannot recreate that with releases.

Senator WHISH-WILSON: So why was the funding halved? It was $3 million to begin with. Why is it $1.5
million over the next three years?

Mr Costello: That was a decision of government. It was an election commitment made in 2016.

Senator WHISH-WILSON: Are you confident as a department, based on what you have seen so far, that
$1.5 million over three years will have any impact or have a measurable impact on the sediment?

Mr Costello: I am happy to report that the Tasmanian government, as part of the city deal, committed half a
million dollars on top of our $1.5 million, so that has taken that up to $2 million. So we were pleased with that
contribution and, in particular, the buy-in of the Tasmanian government into the issue as well. As you know, there
are many players. There is urban stormwater run-off. There is the combined sewage stormwater works. There is
catchment management higher up in the catchment, stabilising riverbanks. There is discharge from dairy. So there
are many, many factors that contribute to the condition.

I understand the scientist said that he would be happy to hear from you about any mistakes they made. Did you
get back to Dr Ritchie about what was erroneous in his article?

Mr Dadswell: As a result of the savings measures from the Green Army program at the mid-year economic
forecast, an additional $100 million over four years was provided for the National Landcare Program. Part of the
government's announcement on that program included $15 million for new Indigenous protected areas. There are
other Indigenous protected areas that are funded under the Natural Heritage Trust.

Ms Campbell: Yes. The Indigenous protected areas funding comes out of the National Landcare Program. In
the four years to 2017-18, we have spent $64.681 million on the Indigenous Protected Areas program. Those are,
as you flagged, for the consultation and ultimate declaration of Indigenous protected areas. We have, I think, 72
Indigenous protected areas around the country. That funding ongoing is a matter for government in terms of the
new National Landcare Program, which was announced in the budget. But on top of the maintenance of those
existing IPAs, the government has announced $15 million for new IPAs, which I envisage will be in identifying
new areas and working to get an IPA on those areas.

Senator CHISHOLM: Sure. So for the next five years from 2018-19, have you got a breakdown of what
would be existing consultation versus new IPAs?

Ms Jonasson: I will also addÔÇöI am sorry to interruptÔÇöthat they are also being worked through with the
Department of the Prime Minister and Cabinet. We have the $15 million, which is for new IPAs. That has been
announced. But the rest of the funding really will come down to decisions there.

Ms Jonasson: I will also add that, as we recall, $15 million was announced in November last year out of the
$100 million that is specifically for new Indigenous protected areas. So that will be supporting some of this, I am
sure.

Senator Birmingham: I think decisions will be made under the process that has been outlined. As, of course,
Ms Jonasson has indicated, there is $15 million additional that was indicated and is under part of a formal
consultation process.

Ms Campbell: The funding for the $100 million, which was provided out of MYEFO, was funding over four
years from 2016-17. That is the funding that the government has announced. Further detail on the funding for the
new Indigenous protected areas is being worked through. We are working very closely with Prime Minister and
Cabinet on developing that to provide advice to government.

Ms Campbell: There is no funding provided under the biodiversity fund for this portfolio. My recollection is
that any contracts under the biodiversity fund were transferred to the Department of the Prime Minister and
Cabinet in 2014 as part of the machinery of government changes. We do have some payments made for the ranger
program under the Landcare program. Those arrangements continue. We pay a relatively small amount of
supplementation money for the ranger program, which is in many ways a legacy of when we managed that
program within this portfolio.

Ms Campbell: Over the four years of the National Landcare Program from 2014 to 2017-18, it is $34.729
million.

Mr Thompson: In November 2016, obviously the Commonwealth, state and territory environment ministers
agreed to pursue a common national approach to environmental economic accounting. For us, that is a framework
for capturing and organising information on the environment and its contribution and its interaction with
economic activity and the impact of that economic activity on the environment. We will be over the course of this
yearÔÇöand we are this yearÔÇödeveloping a strategy on a common national approach to environmental economic
accounts to provide to environment ministers by December 2017. That is the time line they set. As part of that, we
are collaborating with other governments and agencies within the Commonwealth.

Senator WHISH-WILSON: Sure. I might ask some questions about the EPBC status of great white sharks.
The committee does have an ongoing Senate inquiry into this issue and the broader issue of shark mitigation. I
want to ask some questions about specific comments in the media recently. Just confirm that the recovery plan for
the great white that was initiated on 6 August 2013, the EPBC recovery plan, is due for review in 2018.
Mr Knudson: The act itself?

Mr Richardson: Yes, that is correct. Its five-yearly review will be due in 2018.

Mr Richardson: Well, by the deadline in 2018. I am not trying to be cute, but we have not actually scheduled
it as yet. But it will be done. Presumably later this year we will commence that.

Mr Richardson: We have provided some information that CSIRO released, I think, in 2014 on the population
estimates at that time of the east coast population, which is the only available information at this point.
Senator WHISH-WILSON: Is it your understanding that the work being done by CSIRO will update that
2014 report with new population data?

Mr Richardson: That would be a question to ask of 1.2. But my understandingÔÇöthat research project is being
done under our National Environmental Science ProgramÔÇöis that the white shark project, which has CSIRO and
other participants, is looking to refine that east coast population estimate from 2014. So it will refine that for the
east coast. My understanding is it is going to deliver a point estimate of the population size of the west coast
population, but I suspect with bounds around it. It will be a range estimate, if you like.

Senator WHISH-WILSON: That is fine. Did the 2014 study that you provided to the minister show that
population levels of great white sharks have been increasing? I underline the word 'increasing'.

Senator WHISH-WILSON: So unless we get something versus 2014, we do not know whether the
population levels are increasing per se. We know there is a recovery plan in place because, going back over
decades, there were concerns about numbers of great white sharks. That is part of an international treaty on
protection. Mr Frydenberg, our environment minister, said not long ago, on 20 April 2017 that 'Blind Freddy'
could see that there were more 'great white sharks in the water and people's lives would be put in danger if we
don't take action'. Where would the minister have got that idea that blind Freddy could see that white shark
populations were going through the roof?

Mr Oxley: In terms of the steps in the process, each year there is a call for public nominations made. That is
made towards the end of the calendar year. The Threatened Species Scientific Committee then goes through a
process of assessing and looking at all of those nominations. It then puts to the minister what is called a proposed
priority assessment list. The minister considers that list, and the minister may adopt the advice of the Threatened
Species Scientific Committee, which may vary the species that he wishes to have added to what then becomes the
finalised priority assessment list. We are in the midst of that process at the moment. So what is called the FPAL
for 2017 is yet to be determined.

Mr Richardson: There is a current recovery plan, a relatively recent one from 2013.

Senator WHISH-WILSON: Specifically my question is in relation to the recovery plan 2018 and the fiveyear
review that we talked about earlier. Would you expect that you would wait for that review to occur before
this was recommended to the committee or vice versa or the minister used discretion?

Senator WHISH-WILSON: Sorry. That is what I meant. I apologise. He has the discretion to ask the
committee to prioritise an assessment?

Mr Richardson: I think we have covered some of this. My understanding is that there is the National
Environmental Science Programme. Mr Knudson just gave a quantum for some of that, at least. That research on
the east coast has already produced in 2014 a point estimate of the size of the adult population on the east coast.
The end of that research project is the end of this calendar year. I understand that they are going to refine that
population estimate. On the west coast, they are looking to produce their first adult population estimate by the end
of this calendar year.

Mr Johnston: We are hoping and aiming to have it done before 1 February 2018. Whether we get there or not
is yet to be seen, but that is the intent, or the hope, at this point.

Mr Oxley: I might add that if we make that deadlineÔÇöand we certainly expect to be able to do thatÔÇöthat
would enable the World Heritage Committee to be considering that nomination formally at its meeting in the
middle of 2019. So, once submitted, there is a process that runs over the course of about 16 months or thereabouts
for that nomination to be formally assessed and then the proposal finally to be put to the World Heritage
Committee.

Mr Johnston: There are two other places on the tentative list, both of which date back some years. They are
extensions to the Gondwana Rainforests World Heritage site, which straddles New South Wales and Queensland,
and then an extension to the Fraser Island World Heritage listing. The 2015 meeting of environment ministers
discussed the tentative list update and identified that process in their communique. There was mention as well of
Cape York in Queensland, Royal National Park in New South Wales and the West MacDonnells in the Northern
Territory. But the process is that each of those state governments will need to work up a credible case for World
Heritage listing before the Commonwealth would brief the minister about adding it or not to the World Heritage
list.

Mr Johnston: Which is a similar process. They had a discussion in December 2015. We understand that one
of the states has requested another discussion this year on the tentative list. So there may be one later this yearÔÇöa
second discussion on that.

Mr Johnston: Its current due date is 30 June this year, but that can be extended until 30 June 2019. The
minister has the power to extend assessment deadlines for up to five years cumulative. So 2019 would be the final
extension date. With regard to how it has been going, the Australian Heritage Council discussed the assessment in
May 2014. They considered at that time that they did not have sufficient information to assess the Coral Sea for
its historic and natural values as they felt that while it was likely to have outstanding historic values for its
shipwrecks and that some areas of the Coral Sea may meet the criteria for natural heritage value, it was unlikely
the entire Coral Sea would meet the national heritage criteria. Following that discussion, they did not continue to
do any more work on it at that point. But, in recent months, the department has decided to scope out a shape of
how an assessment might look. I am checking if we have it scheduled. In fact, the upcoming Heritage Council
meeting is scheduled at this stage to consider a scoping for that assessment.

Senator RICE: I understand that there was an emergency intervention two years ago in 2015 and that a oneyear
timeline was established for that recovery plan. Is that the case?

Senator RICE: I have asked questions about the recovery plan before. It was in draft a year agoÔÇöearly in
2016ÔÇöor over a year. Is that the case?

Senator RICE: So you are confident that you will get in by April 2018. Do you expect to come in earlier than
that?

Mr Richardson: That is another potential complication. I am not saying it is a complication at this point. As
you would be aware, the Victorian government is reviewing their forest prescriptions, or at least their buffers.
There has been a lot in the media about this but also on their website about the review that they are undertaking of
the, I guess, efficacy of the arrangements that were brought in in 2014 out of the Leadbeater's possum advisory
group process that the Victorian government ran. So that is when they introduced their 200-metre buffers around
newly discovered colonies. So they are conducting a review of at least those buffers; I am not sure if it is broader
than that. I understand that their website says that that review will be reported to the government by the end of
April. Presumably, the Victorian government will act on it and make some announcement at some point.

Mr Knudson: In 2015, a comprehensive conservation advice was prepared for the species. It was approved
back then as well. That has already been taken into account through individual assessments where the species is
potentially impacted.

Mr Richardson: As I mentioned earlier, at the time of listing, and, in this case uplisting of the species, a new
conservation advice was prepared. So this was done in 2015, as Mr Knudson said, at the time of the uplisting to
critically endangered for the possum. So the then minister made that decision in April 2015 and a new
conservation advice was published.

Senator WHISH-WILSON: I will go to that first before I lose my train of thought. I have a couple of other
quick questions. So that is October 2018, which will precede the whaling fleet going down there again for
summer?

Senator WHISH-WILSON: Okay. This is the last question for me. I am very hopeful, based on what you
said, that maybe we will get a clear statement. Is there any other process or intergovernmental process going on at
the moment in relation to potential court action or other representations to the Japanese government on this issue,
or are you waiting on today's outcome and potentially a statement by 2018?

Dr de Brouwer: There are two major programs. The $28.7 million is for the industry portfolio. That is really
work for geoscience to assess gas. The other is, though, $30.4 million for bioregional assessments, which is
looking at the interaction of gas reserves or coal seam gas with water.

Senator URQUHART: So $28.7 million for industry for geoscience?

Dr de Brouwer: We do a lot. In that respect, there are extensive bioregional studies as well as support for the
independent expert scientific committee, which provides advice around coal seam gas and water. In the energy
area, there is a lot of discussion around the impact of state policy, particularly moratoriums and bans on the
supply of gas. That industry measure that we mentionedÔÇösupporting the development of new onshore gas supply,
the $28.7 million programÔÇöis designed to provide an incentive for onshore gas exploration and development. But
that is, again, provided through the industry department. Those issues are going to come up in the array of policies
that the government has got in increasing the gas supply domestically and the efficiency of the domestic gas
market. But 4.1 is the best place to raise that.

Senator URQUHART: I will follow that up in 4.1. In relation to Abbot Point, it was reported yesterday that
the government has revoked a number of conditions from the Abbot Point approval and replaced them with one
condition to provide the Reef Trust with $450,000. Can you confirm that the reports about conditions being
removed are correct?

Dr de Brouwer: They are incorrect. We can explain that.

Mr Knudson: I will turn to my colleagues if they feel they want to add anything more to this. What has
happened here is this is the exact same amount of money being required for the exact same period of time and for
the same purpose. We have changed the conditions to allow that funding to go through the Reef Trust. The Reef
Trust in turn will then develop the projects with the advice of the independent expert panel that has been set up
under the Reef 2050 Plan to make sure that those projects are as scientifically robust as possible, are focused on
the key threats in and around the reef and that address the issues that are most important. Whereas under a regular
project approval, a proponent could come in and say, 'I'm going to spend $450,000 on this matter,' but it is not
done in a linked up way and an integrated way as the Reef Trust allows. That is what has changed in these
conditions.

Mr Knudson: We talked about that a fair amount when Ms Parry was at the table. This is the $210 million
fund that has been set up. Most recently, its phase 4 and phase 5 investments have been in dealing with nutrient
run-off, gully erosion and a range of other activities in and around the reef. They are looking at those onshore and
marine environment investments. So they have built up a fair amount of experience over the last several years in
designing projects to deliver the types of environmental outcomes that we are looking for. This is absolutely a
good news story that these offsets will be going through the Reef Trust.

Ms Collins: Part of the establishment of the Reef Trust is that it is specifically able to accept and implement
offsets from developments that are approved under the EPBC Act. In fact, this is not the first time that the Reef
Trust has been engaged to provide those offsets. Since its establishment, the Reef Trust has provided more than
$5 million worth of offsets.

Ms Collins: The payments are required to be made annually until the year 2053. If Adani was to commence
the activity this year then the first payment would be made this year?

Ms Collins: But the payments are required to be made continually each year until 2053.

Ms Collins: As I said earlier, Adani did not specifically ask for it. It is a position that the department took up
and it is in line with the department's commitment to the Reef Trust, and it is in line with the view of getting the
broader strategic outcomes at a larger scale, consistent with the Reef 2050 Plan, and it takes away from this siteby-site
decisions made by individual companies. So the objective is to the broader environmental outcome that is
consistent with the Reef 2050 Plan.

Senator WATERS: I have two further questions before I move on to other issues. My understanding is that
under the Reef Trust architecture the government is free to use the $450 million, if and when you receive it, on
whatever projects you like, as opposed to the earlier marine offset strategy, which was specified in the condition
that required it to be spent on particular mattersÔÇöturtles and the like.

Senator WATERS: I ask now about the requirement for Adani to publish management plans on their website
within one month of them being approved. I asked you this and you responded in question on notice No. 77
saying that the biodiversity offset strategy was approved on 7 October 2016. However, it was not published on the
company's website as it was meant to be until approximately March. Can you tell me what date it did go online
and why the delay.

Senator WATERS: I understand that the funds under that program were meant to flow in the third quarter of
2016 and that has not occurred yet. I am interested in why and what the hold-up is. Is it coming or are they trying
to amend their conditions in the manner, perhaps, that Adani has benefited from, a deferral of payment falling
due. On a related issue, I am interested in whether they have identified a final site for the marine reserve. Again,
still can't help?

Senator WATERS: Under their Coastal Offset Strategy they were meant to have announced those final sites
in January 2014, and that was obviously three years ago. So why has the department let the company default on
their obligations for more than three years? I am keen to get a response on that when you can.

Mr Barker: With regard to the question you asked in relation to the original Toondah Harbour referral: in
November 2015 there was advice sought, as is the department's ordinary practice, from a number of areas within
the department that included the department's wetlands area.
'''
