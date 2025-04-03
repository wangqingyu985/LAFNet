"""
Created by Mr. Qingyu Wang at 14:33 07.03.2025
E-mail address: 12013027@zju.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt


x = [1, 3, 5, 7, 9, 11, 13]
labels = ['1', '2', '4', '8', '16', '32', '64']
model_1 = np.array([0.14492005109786987, 0.9793421626091003, 0.5591928958892822, 0.9765912294387817, 0.8023801445960999, 1.673004388809204, 0.31146442890167236, 2.4472837448120117, 0.378044992685318, 2.6443099975585938, 0.18116824328899384, 0.9657366275787354, 0.417284220457077, 1.4898920059204102, 0.02214759588241577, 0.034120529890060425, 0.40722280740737915, 0.7732387781143188, 0.9878157377243042, 2.1537864208221436, 0.6813644170761108, 1.5935828685760498, 0.8522570133209229, 1.0265603065490723, 0.0360165536403656, 2.1411654949188232, 0.2992580831050873, 2.048384666442871, 0.5155903697013855, 0.4837435483932495, 0.43141961097717285, 0.5310524106025696, 2.1393280029296875, 0.5701630711555481, 0.33882856369018555, 2.5951366424560547, 1.774477481842041, 0.5912401676177979, 0.42513325810432434, 1.7756390571594238, 2.013180732727051, 1.7189967632293701, 0.04037253558635712, 0.04924684762954712, 0.6432123184204102, 1.639851450920105, 1.6436480283737183, 0.025028660893440247, 0.0484466552734375, 0.6939465999603271, 0.5050463676452637, 1.069847583770752, 0.04695713520050049, 1.8538187742233276, 0.018346980214118958, 0.4295197129249573, 0.4323139190673828, 0.22086754441261292, 1.698835849761963, 1.5501688718795776, 0.5580052733421326, 1.9740586280822754, 0.3726990520954132, 0.5375292897224426, 1.8138563632965088, 0.9324613213539124, 1.486058235168457, 2.1878957748413086])
model_2 = np.array([0.09394899010658264, 0.615612268447876, 2.411525011062622, 0.6795957088470459, 0.45281529426574707, 0.034125447273254395, 1.467212438583374, 0.08817553520202637, 0.6858772039413452, 0.9617835283279419, 0.2807101607322693, 0.17717979848384857, 0.26941269636154175, 0.18086418509483337, 0.06769396364688873, 0.7201692461967468, 0.133746936917305, 0.09020194411277771, 0.8977770209312439, 0.7023090124130249, 0.09711779654026031, 1.5110303163528442, 1.3442147970199585, 0.9578193426132202, 1.1559654474258423, 0.8888670802116394, 0.2564460039138794, 0.24079382419586182, 0.7363560795783997, 0.13102814555168152, 0.8327637910842896, 0.6894716620445251, 0.5650243163108826, 1.6005299091339111, 0.4212471842765808, 0.43084055185317993, 1.9440832138061523, 0.7915687561035156, 0.7388284206390381, 0.21899938583374023, 0.7528209686279297, 1.921360731124878, 0.01342684030532837, 0.21457239985466003, 1.0392893552780151, 0.14407113194465637, 1.580819845199585, 1.4827213287353516, 1.7176077365875244, 0.0038813799619674683, 3.270176649093628, 2.2709527015686035, 0.4217666983604431, 0.3873887062072754, 1.1498260498046875, 0.14743536710739136, 0.17552852630615234, 1.158788800239563, 0.18233813345432281, 0.23428939282894135, 1.316227912902832, 1.6980643272399902, 0.7159907817840576, 0.023091137409210205, 0.3933835029602051, 2.4492244720458984, 0.4048626124858856, 2.8243253231048584])
model_4 = np.array([1.5608525276184082, 0.6138777732849121, 2.9213876724243164, 0.8987407684326172, 0.17865456640720367, 0.16880780458450317, 0.2525704503059387, 0.3442481756210327, 0.14420658349990845, 0.8531028032302856, 0.20450502634048462, 0.3103896379470825, 0.21996058523654938, 0.359245240688324, 0.101136714220047, 0.2886424958705902, 1.7604708671569824, 0.5542212724685669, 0.5276826024055481, 0.7611446380615234, 0.3354984521865845, 0.7368278503417969, 0.4963850975036621, 0.8025685548782349, 1.005108118057251, 0.9054619073867798, 1.5810201168060303, 1.642230749130249, 1.3987271785736084, 0.27030736207962036, 0.45765548944473267, 0.16619868576526642, 0.01458287239074707, 1.674246907234192, 1.0725733041763306, 1.121969223022461, 1.1920785903930664, 0.3310198187828064, 0.19601672887802124, 0.1437550038099289, 0.36257535219192505, 1.2051182985305786, 0.35382822155952454, 0.4156178832054138, 0.36115431785583496, 0.933985710144043, 0.6018967032432556, 0.31954681873321533, 0.2978280186653137, 1.1775403022766113, 1.246338129043579, 0.18013454973697662, 0.28318077325820923, 0.8720033764839172, 0.12972578406333923, 0.3267764449119568, 0.14574728906154633, 0.277457594871521, 1.429290771484375, 0.20759224891662598, 0.46310752630233765, 1.2773871421813965, 1.302191138267517, 1.9988871812820435, 2.8757400512695312, 0.41695699095726013, 0.38954973220825195, 0.007461607456207275, 0.8611629009246826, 0.4461570978164673, 0.6991615295410156, 1.061469316482544])
model_8 = np.array([0.9118386507034302, 2.4705142974853516, 0.05622401833534241, 0.9445176124572754, 0.7320997714996338, 0.3225645422935486, 1.6045022010803223, 3.086972713470459, 2.1674835681915283, 1.0366147756576538, 1.2970149517059326, 0.17402037978172302, 0.7677496671676636, 3.0127012729644775, 0.6627312302589417, 0.6890213489532471, 3.2251179218292236, 0.6860511302947998, 1.249481439590454, 1.4767528772354126, 0.7632936239242554, 0.06855368614196777, 2.4110965728759766, 3.330585479736328, 0.508244514465332, 1.441102147102356, 3.4018874168395996, 3.4018869400024414, 0.5404772758483887, 0.220068097114563, 1.0470129251480103, 0.5409239530563354, 0.9063428640365601, 0.4246130585670471, 0.9252066016197205, 1.1217310428619385, 0.6073223352432251, 0.8524202108383179, 3.3885180950164795, 0.8494490385055542, 0.8197401762008667, 0.9375372529029846, 0.8394977450370789, 0.6444584131240845, 0.24784520268440247, 0.5226525664329529, 1.1321296691894531, 0.8900028467178345, 0.5033419132232666, 2.5982608795166016, 0.09633049368858337, 0.8093422651290894, 3.3513827323913574, 3.271167039871216, 0.993537187576294, 0.1884271204471588, 0.5810315608978271, 0.6637687087059021, 0.5627591609954834, 0.25423452258110046, 0.9890807867050171, 0.6137107610702515, 1.4901227951049805, 0.4142146706581116, 0.6503999829292297, 0.47363221645355225, 0.17484545707702637, 1.634866714477539, 0.4639431834220886, 2.4204158782958984, 0.15840619802474976, 0.6897565126419067, 1.3287594318389893, 0.6033503413200378, 0.18848009407520294, 2.2078733444213867, 0.6082623600959778, 1.09136164188385, 1.129900336265564, 0.2696155905723572, 0.20434267818927765, 0.2441055327653885, 0.9516420960426331, 1.5151065587997437, 1.2051278352737427, 0.8226830363273621, 0.2884315848350525, 0.16960658133029938, 0.104714035987854, 0.01507878303527832, 1.6548893451690674, 0.3285515308380127, 0.6079408526420593, 0.39839237928390503, 0.17577864229679108, 1.919750690460205, 0.31750863790512085, 0.2950948476791382, 0.11405795812606812, 0.6677114963531494, 0.44160306453704834, 0.4956268072128296, 0.20073629915714264, 1.7031277418136597, 0.8467963337898254, 2.071394443511963, 0.4800521731376648, 0.013306587934494019, 0.03277525305747986, 0.19928663969039917, 0.6273314952850342, 0.151209756731987, 0.9504646062850952, 2.6099815368652344, 0.49267417192459106, 0.6152937412261963, 1.3101683855056763, 0.5469714999198914, 0.06279699504375458, 0.4354691505432129, 0.4273010790348053, 1.585734486579895, 0.14980383217334747, 2.562915086746216, 0.35897696018218994, 0.09035363793373108, 0.032173991203308105, 0.015861988067626953, 1.583646535873413, 0.37709712982177734, 1.156419277191162, 2.1584365367889404])
model_16 = np.array([0.056116729974746704, 0.6829735040664673, 0.6458374261856079, 2.385951042175293, 0.9102462530136108, 0.5388855338096619, 0.16901031136512756, 0.07394194602966309, 1.6565985679626465, 2.2968246936798096, 0.23436978459358215, 2.3161354064941406, 0.5180894136428833, 1.0365087985992432, 0.9607512354850769, 1.2429853677749634, 0.07757321000099182, 0.7215949892997742, 3.2593913078308105, 0.9934309720993042, 0.3338944911956787, 0.07691293954849243, 2.510728359222412, 0.5284874439239502, 0.25285542011260986, 0.7587310671806335, 0.42153552174568176, 0.16372889280319214, 1.0439361333847046, 0.5210602283477783, 3.2207698822021484, 0.6413810849189758, 1.1122664213180542, 0.8641974925994873, 0.18749594688415527, 0.9206443428993225, 3.2593913078308105, 1.1479170322418213, 0.8529741764068604, 0.8092361092567444, 0.6970027089118958, 0.7973525524139404, 0.9949164390563965, 2.0249884128570557, 0.7950420379638672, 0.030864179134368896, 2.0576682090759277, 0.013038814067840576, 0.6725753545761108, 0.9518386125564575, 0.06568968296051025, 0.7364494204521179, 1.1791112422943115, 3.0662841796875, 0.9354987740516663, 0.44744834303855896, 3.2593913078308105, 1.10335373878479, 0.047204047441482544, 0.5306332111358643, 0.48458442091941833, 0.7928962111473083, 2.374067544937134, 1.106985092163086, 2.4557669162750244, 0.6042450070381165, 2.880809783935547, 0.5312519669532776, 3.2447433471679688, 2.22721529006958])
model_32 = np.array([1.2699686288833618, 3.16853404045105, 0.7041847705841064, 1.075545310974121, 1.6801204681396484, 0.99384605884552, 1.203123688697815, 0.8258212804794312, 0.17833799123764038, 0.3937271237373352, 1.2268908023834229, 1.082972526550293, 0.8406757116317749, 0.9149478673934937, 0.8273067474365234, 0.4396061599254608, 0.14697402715682983, 2.2683558464050293, 0.43977588415145874, 1.9668108224868774, 2.566929817199707, 0.8987775444984436, 0.9446566700935364, 0.18725067377090454, 0.07287159562110901, 0.888209879398346, 0.1813088357448578, 2.798658847808838, 2.1005008220672607, 0.8377048373222351, 0.9669383764266968, 2.779348134994507, 1.380061149597168, 2.488201141357422, 1.1511332988739014, 1.3338425159454346, 0.4648587107658386, 2.969484567642212, 1.1347934007644653, 2.8536200523376465, 0.4264068603515625, 1.0278414487838745, 1.3071047067642212, 0.09663864970207214, 0.16479924321174622, 2.177743911743164, 0.292547345161438, 0.04316270351409912, 2.283210277557373, 0.9223750233650208, 0.9505984783172607, 0.955054759979248, 0.4128682613372803, 2.5654444694519043, 2.309948205947876, 1.0233851671218872, 0.8124523162841797, 2.23716139793396, 0.46948477625846863, 0.3149986267089844, 0.34767836332321167, 0.209532231092453, 0.08921146392822266, 0.7842289209365845, 1.1377642154693604, 0.19005179405212402])
model_64 = np.array([3.3231399059295654, 2.772040605545044, 2.5002048015594482, 1.288083553314209, 3.373662233352661, 0.8405386209487915, 3.373662233352661, 2.984476327896118, 0.7499265670776367, 1.7946364879608154, 2.518047332763672, 0.2240799367427826, 1.6639175415039062, 3.0557775497436523, 2.570037841796875, 0.6578291058540344, 0.5126996040344238, 1.0648404359817505, 0.19184410572052002, 3.373662233352661, 2.4883384704589844, 2.7720580101013184, 0.552806556224823, 2.4422895908355713, 0.558304488658905, 3.118165969848633, 0.3894078731536865, 0.9093129634857178, 0.6786253452301025, 0.9890828132629395, 0.739528477191925, 0.5617192387580872, 0.45283809304237366, 2.608659267425537, 0.8123152256011963, 0.6449040770530701, 0.30176684260368347, 1.448528528213501, 0.22600919008255005, 0.3596991300582886, 0.7053632736206055, 2.7126402854919434, 0.041370391845703125, 0.8008755445480347, 0.02993077039718628, 1.8867340087890625, 0.7900335788726807, 0.09380489587783813, 0.6924381852149963, 0.9014416933059692, 0.782606303691864, 3.1805548667907715, 2.4363479614257812, 0.884060263633728, 0.3028082847595215, 0.17996057868003845, 0.861778736114502, 0.04032894968986511, 0.3864370286464691, 0.8836164474487305, 0.13242632150650024, 0.7959753274917603, 0.7454702854156494, 0.5854862928390503, 0.12009888887405396, 0.41466042399406433, 0.8687620162963867, 3.196894645690918, 0.45432353019714355, 2.41109561920166])

all_data = [model_1, model_2, model_4, model_8, model_16, model_32, model_64]

plt.figure(figsize=(6, 4.5), dpi=600)

axis_font = {'weight': 'bold', 'size': 14}
title_font = {'weight': 'bold', 'size': 15}
plt.rcParams.update({'font.size': 13})
plt.rcParams["font.weight"] = "bold"

colors_pale = ['#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6', '#ffffcc', '#e5d8bd', '#fddaec', '#f2f2f2']

box = plt.boxplot(all_data, patch_artist=True, positions=[1, 3, 5, 7, 9, 11, 13],
                  showmeans=False, widths=0.8, sym="k+")
for patch, color in zip(box['boxes'], colors_pale):
    patch.set_facecolor(color)

plt.ylabel("Error (N)", fontdict=axis_font)
plt.xlim(0, 14)
plt.ylim(0, 5)
plt.xticks(ticks=x, labels=labels)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
plt.show()
