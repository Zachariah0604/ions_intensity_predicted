
data
--data_swedcad_mm SwedCAD��MMģ����������
--pre_data ����Ԥ��������
----MMdata MMԤ��������
------3result ���ս��
------comet comet����������䴦����
------database ���ݿ�
------mascot mascot����������䴦����
------mgf mgfԴ�ļ�
------pFind pFind����������䴦����
----ProteomeToolsData ProteomeToolsԤ��������
------InitialFile ��ʼ�ļ�
------raw_to_mgf ��ȡ��������mgf
------result ����ļ���
data_preprocessing
--mm_data.py MM���ݴ���ű�
--proteometools_data.py ProteomeTools ���ݴ���ű�
improved-lstm-WGAN improved-WGAN��lstm���ģ��
lstm LSTMģ��
lstm-GAN lstm��GAN���ģ��
seq2seq-WGAN seq2seq��WGAN���ģ��


ProteomeTools���ݴ�����⣺
1����ʼԴ�ļ�����raw�ļ���InitialFile/raw/HCD/*.raw������01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1.raw��������Ӧ�ļ��������InitialFile/identification_results/HCD/*.zip������TUM_first_pool_1_01_01_2xIT_2xHCD-1h-R1-tryptic.zip��
2������SpectrumAnalyzer����raw�ļ�������������������ͬĿ¼��InitialFile/raw/HCD/*-analysis.txt������01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1-analysis.txt��
3���������ȣ��ڲ�����ʡ�ԣ���ѹ��InitialFile/identification_results/HCD/�µ�����ѹ���ļ�����ȡÿһ����ѹ���ļ��е�msms.txt��ͨ��pif>=0.7��score>=100����msms.txt��ͬʱ��-analysis.txt��Ѱ����ͬSpectrumId��ͼ�׵�NCE�������������InitialFile/identification_results/HCD/msms.txt����msms.txt�а���Scan number,Raw File Name,Sequence,Charge,PEPMZ,PIF,Score,Modifications,NCE
4��ͨ��pParse.exe���˲�����ͨ��Դ���Զ����õ����е�pParse.exe���򣩷���.raw�õ���Ӧ��*_CIDIT.mgf,*_HCDFT.mgf��*__HCDIT.mgf�ļ�������01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1_CIDIT.mgf�ȣ���
5���������Ĳ��е�.mgf�ļ�������ϵ������е�msms.txtд���µ�mgf�ļ�������ڣ�raw_to_mgf/HCD/*.mgf����01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1.mgf��
6����ȡ5�е�����mgf�ļ�����NCE�ֵ�ɴ�Ž���ڣ�result/mgf_result/HCD/NCE*/mgf�ļ���/charge*.txt�������磺result/mgf_result/HCD/NCE20/mgf_01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1/charge2.txt�����ý���ļ�������NCE,SCORE,SpectrumName,Charge,SEQ,Modification,PepMass
7����ȡ6�еĽ����5����Ӧ��mgf�ļ�ƥ��b,y,by,ay���Ӳ����������ڣ�result/mgf_result/HCD/NCE*/mgf�ļ���/all_charge/match_ion/ay_by.txt��b_y.txt��

ʵ����֤�ļ������ data/pre_data/ProteomeToolsData/��

���У�01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1.raw��TUM_first_pool_1_01_01_2xIT_2xHCD-1h-R1-tryptic.zipΪԭʼ�ļ�
01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1.mgfΪʵ�鴦�����õ�mgf�ļ�
NCE*_Charge2_*.txtΪ����ƥ��Ľ�����˴��о���NCE20��NCE23��2�۵�ay,by,b,y��ƥ������

