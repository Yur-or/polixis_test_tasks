import requests as r
import pprint


def main():

    # Test simple text prediction
    text = "This movie was exactly what I wanted in a Godzilla vs Kong movie. It's big loud, brash and dumb, in the best ways possible. It also has a heart in a the form of Jia (Kaylee Hottle) and a superbly expressionful Kong. The scenes of him in the hollow world are especially impactful and beautifully shot/animated. Kong really is the emotional core of the film (with Godzilla more of an indifferent force of nature), and is done so well he may even convert a few members of Team Godzilla."

    keys = {'text': text, 'model_name': 'bert'}

    prediction = r.get("http://127.0.0.1:8000/predict/", params=keys).text

    res = eval(prediction)

    print('Result for simple text:')
    pprint.pprint(res)


    # Test batch texts prediction
    texts = ["European Union regulators believe their rescue of Spanish lender Banco Popular POP.MC has strengthened the case for intervening in Italy's two weakest lenders, but expect it will be harder to use the same approach, a senior EU official said on Wednesday. EU regulators arranged for Spain's biggest bank, Santander (SAN.MC), to take over Banco Popular, but only after wiping out the investments of the troubled lender's shareholders and junior creditors -- a move welcomed by financial markets which saw it as a possible template for other EU banking crises. Italy is struggling to resolve a crisis inside two regional banks, Banca Popolare di Vicenza and Veneto Banca, which are in an even weaker position in terms of capital than Banco Popular, according to the EU official who declined to identified. Unlike Banco Popular, however, the official said the two Veneto banks lack a willing buyer like Santander. Without a buyer, the banks, which face a combined capital shortfall of 6.4 billion euros ($7.2 billion), run the risk that regulators would wind them down and impose losses on senior creditors and large depositors -- which Banco Popular avoided. The Italian government is opposed to such a solution and will explore every alternative option, the official said. Financial markets, too, could react badly if senior creditors were to be hit in Italy. The two Veneto banks declined to comment.  Italy is home to the euro zone's fourth-largest banking industry, which holds a third of the bloc's total bad debts. However, both regulators and Rome have so far shown less willingness to take swift, decisive action there. Apart from the Veneto banks, which have been in crisis for two and a half years, Italy has also been propping up its fourth-largest lender, Monte dei Paschi di Siena (BMPS.MI). The EU last week gave a preliminary green light to a state bailout of the world's oldest lender after months of negotiations.   Stronger Hand The EU official said the  relatively successful  resolution of Banco Popular could in principle strengthen the case for winding down the two Veneto banks.  The resolution did not trigger negative reactions in the market because of the intervention of a solid bank,  the official added, noting that losses had been limited to ordinary shareholders and junior bondholders.  In Italy it is difficult to find buyers and the two (Veneto) banks are in worse solvency conditions. Popular went bust because of liquidity problems, not solvency.  A second source familiar with the two Veneto banks' situation said Italy cannot take much longer to resolve their worsening crisis.  The longer you wait, the smaller the chances of success,  the source said.    Spain has delivered a colossal lesson - a solution in a few days, within ECB rules and with a united banking sector stepping in.   The banks have already been bailed out last year by a special banking-rescue fund orchestrated by Rome and financed by dozens of private investors, including Italy's healthier banks. They have requested state aid like Monte dei Paschi, but negotiations have been trickier with Brussels, which has demanded an injection of additional private capital before any taxpayer money can be used. The two lenders have collectively lost 7.5 billion euros in direct funding in 2016 and are still bleeding deposits but continue to operate thanks to government-guaranteed liquidity. The EU has agreed that Rome can step in to provide capital as well, but it first requires that private investors stump another 1 billion euros. No bank has shown a willingness to invest more money in the two Veneto banks. Heavyweights Intesa SanPaolo (ISP.MI) and UniCredit (CRDI.MI) are the only two Italian lenders with a size that would allow them to take on a Veneto bank. However, Intesa has pointedly ruled out investing any more money in the two lenders and has always said it does not want to grow its domestic market share any further. UniCredit, meanwhile, has just completed a 13 billion euro share issue to clean up its balance sheet. writing by Silvia Aloisi, editing by Mark Bendeich and Susan Fenton",
            "Brazilian petrochemical company Braskem may be suspended from Level 1 trading at the São Paulo Stock Exchange after failing to submit certain financial statements within a deadline, stock market operator B3 told Reuters on Wednesday. Braskem, whose stock market quotes are being given separately as a result of the delay, had until June 22 to formally present a defense.  The case will be ruled on this week and could result in the suspension of the company,  the bourse said. If the problem persists, at the end of the process Braskem may be compulsorily delisted from Level 1 trading, the bourse said in an emailed statement. Local newspaper Valor Economico said on Wednesday a ruling on the suspension would be made this week. It said Braskem had submitted only unaudited statements for the first quarter, the fourth quarter of 2016 and the full 2016 result. Braskem said it is working with independent auditors to finish the financial statements. According to Valor, Braskem said the delay was due to a pending analysis by independent auditors related to internal controls and procedures after the company signed plea deals with federal prosecutors in Brazil, the United States and Switzerland in connection with a massive graft scheme known as Operation Car Wash. Brazil-based construction firm Odebrecht SA and Braskem agreed to pay at least $3.5 billion in the leniency deal it signed to resolve some international charges involving payoffs to former executives at Brazil's state-controlled Petroleo Brasileiro SA, politicians and others. Braskem is jointly owned by Odebrecht and Petrobras. According to B3, other listed companies have been suspended from trading due to delays in submitting financial statements, including mining concern MMX Mineração e Metálicos in 2014 and engineering firm Inepar SA Industria e Construções in 2016. Level 1 trading requires companies to maintain a free-float of at least 25 percent of capital. The deadline to cure delays in the submission of financial statements is 30 days, B3 said. Braskem's shares closed flat on Wednesday in Sao Paulo, at 34.34 reais. (Reporting by Ana Mano; editing by Jonathan Oatis)", 
            "France is considering creating a special court that would handle disputes over financial contracts governed by English law after Britain leaves the European Union, a move designed in part to help lure banks to Paris from the City of London. Banks in Britain and throughout the European Union generally place their loan and derivatives contracts under English law, but under EU reciprocity rules any judgement is instantly enforceable in any EU jurisdiction. A British exit from the EU's single market would also mean leaving the so-called Brussels convention that requires recognition and enforcement of foreign judgements in all EU states. The vast majority of derivatives and the contracts that underpin them are now handled in London, where an average of 885 billion euros of euro-denominated derivative trades are cleared every day, according to figures from the Bank of International Settlements. Banks are already examining reworking contracts so their EU entities would be named as the involved bank not their UK ones. That task would become more complicated if they had to change the law they were written under or find a workaround so they could still be enforced in the EU but arbitrated in London. France's financial markets law committee has sent a proposal to the Justice Ministry suggesting the establishment of special courts which would rule on matters under English common law and in English. Under EU codes, rulings would be enforceable instantly across the EU.  We would need to recruit judges that have experience in common law,  Christian Noyer, a former Bank of France chief now tasked with winning business from London, told Reuters. Banks have been developing plans for how to cope with a  hard  Brexit, which would see Britain leave the single market and customs union, and have considered varying degrees of upheaval to their British-bases businesses. The Brexit process may have been complicated by May's botched election gamble, which has led to calls in her party for a more business-friendly and softer tack to Britain's departure.  The enforcement of judgements rendered in the UK will not be as swift, cost-free and automatic as it is today unless new rules are agreed between the UK and the EU,  said Valentin Autret, a litigation counsel at Skadden, based in Paris.  Companies will consider strongly bringing action before courts in EU member states in order to benefit from EU rules on the recognition and enforcement of judgements,  Autret added. In the race to win new business after Britain's exist, Paris faces competition from European financial centres such as Frankfurt, Amsterdam and Dublin.  A bank needs to know that if it submits a case to a judge, the decision will be implemented as soon as possible,  said Gérard Gardella, secretary general of France's financial markets law committee. The Justice Ministry did not respond to emails and calls for comment. The committee proposes creating an international court within the Paris Commercial Court and a parallel court at the Paris Appeals court. Hearings, the examination of evidence, the exchange of papers and debates would all take place in English. Some lawyers have, however, expressed scepticism.  Any new French system is likely to rely on French judges only,  said Erwan Poisson at law firm Allen Overy.  Resorting to foreign judges can obviously pose a problem in terms of sovereignty. But not doing so could make the new system less popular,  he said. Other options are available to companies. The Netherlands is launching an international commercial court this year which would settle Dutch legal proceedings in the English language. Depending on the assets involved and the cost of litigation, banks could also opt to seek arbitration. (Additional reporting by Olivia Oran in New York, Huw Jones and Anjuli Davies in London; Editing by Richard Lough and Edmund Blair)"]

    keys_batch = {'texts': texts, 'model_name': 'bert'}

    prediction_batch = r.get("http://127.0.0.1:8000/predict_batch/", params=keys_batch).text

    res_batch = eval(prediction_batch)
    
    print('\nResult for batch texts:')
    pprint.pprint(res_batch)


if __name__ == '__main__':
    main()