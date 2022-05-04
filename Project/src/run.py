import os
import xml.etree.ElementTree as etree
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp
from lmfit import Model
import pandas as pd
import statsmodels.api as sm
from dateutil.parser import parse
projectpath = os.path.dirname(os.getcwd())
datpath ='{}/dat'.format(projectpath)   # dat폴더의 경로입니다.
filepath=[]
for (root, directories, files) in os.walk(datpath):       # 분석해야하는 모든 파일의 경로를 알아냅니다.
    for file in files:
        if '.xml' in file:                                # xml파일만 골라냅니다.
            file_path = os.path.join(root, file)
            filepath.append(file_path)
c=0                                                       # 순회횟수를 알기위한 임시변수입니다.
dict = {'Lot': [], 'Wafer': [], 'Mask': [], 'TestSite': [], 'Name': [], 'Date': [], 'Operator': [], 'DieRow': [],
            'DieColumn': [], 'AnalysisWavelength (nm)': [], 'Rsq of Ref. spectrum (6th)': [],
            'Max transmission of Ref. spec. (dB)': [], 'Rsq of IV': [], 'I at -1V[A]': [], 'I at 1V[A]': []}   # 엑셀파일을 만들기위한 딕셔너리입니다.
for allfiles in filepath :                                # dat의 모든파일을 순회합니다.
    c=c+1
    filename=os.path.basename(allfiles)
    xml = etree.parse(allfiles)
    root = xml.getroot()
    # I-V graph
    V = []
    for v in root.iter('Voltage'):
        V.extend(list(map(float, v.text.split(','))))
    I = []
    for i in root.iter('Current'):
        I.extend(list(map(float, i.text.split(','))))
        I = list(map(abs, I))

    plt.figure(c, [18, 8])
    plt.subplot(2, 3, 4)
    plt.plot(V, I, 'b.', label='data', markersize=8)
    plt.yscale('log')
    plt.title('I-V analysis', fontsize=12, fontweight='bold')
    plt.xlabel('Voltage[V]', fontsize=10)
    plt.ylabel('Current[A]', fontsize=10)

    # IV fitting
    x = np.array(V[:])
    y = np.array(I[:])
    fit1 = np.polyfit(x, y, 12)
    fit1 = np.poly1d(fit1)
    def IV_fit(X, Is, Vt):
        return (Is * (exp(X / Vt) - 1) + fit1(X))
    model = Model(IV_fit)
    result = model.fit(I, X=V, Is=10 ** -15, Vt=0.026)
    initial_list = []
    for i in V:
        x_value = IV_fit(i, 10e-16, 0.026)
        initial_list.append(x_value)
    initial = sm.add_constant(np.abs(y))
    result1 = sm.OLS(initial_list, initial).fit()
    IVdic = {I: V for V, I in zip(result.best_fit, V)}

    # R-squared
    def IVR(y):                     # 결정계수 공식
        yhat = result.best_fit
        ybar = np.sum(y) / len(y)
        sse = np.sum((yhat - ybar) ** 2)
        sst = np.sum((y - ybar) ** 2)
        return sse / sst
    plt.plot(x, result.best_fit, label='best_fit')
    plt.plot(x, result.best_fit, 'r-', label='R-squared ={}'.format(IVR(y)))     # y인이유?
    plt.legend(loc='best', fontsize=8)
    plt.text(1.0, IVdic[1.0], IVdic[1.0])
    plt.text(-1.0, IVdic[-1.0], IVdic[-1.0])

    # wavelength-transmission graph
    L = []
    IL = []
    DCBias = []
    for i in root.iter('WavelengthSweep'):                   # L과 IL, DC bias 데이터 수집
        L.append(list(map(float, i[0].text.split(','))))
        IL.append(list(map(float, i[1].text.split(','))))
        DCBias.append('DC = {}'.format(i.attrib['DCBias']))
    DCBias[-1] = 'Reference'
    def poly(x, y, deg):                        # 결정계수 구하기
        coeffs = np.polyfit(x, y, deg)
        # r-squared
        p = np.poly1d(coeffs)
        yhat = p(x)
        ybar = np.sum(y) / len(y)  # or sum(y)/len(y)
        ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
        sstot = np.sum((y - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
        results = ssreg / sstot
        return results
    Rref = poly(L[-1], IL[-1], 6)                # 6차 결정계수
    fit_L = np.polyfit(L[-1], IL[-1], 6)         # IL fitting
    fit_IL = np.polyval(fit_L, L[-1])

    for i in range(1, 4): # 그래프 그리기
        plt.subplot(2, 3, i)
        if i==1 :                                             # raw data
            for r in range(len(L)):
                plt.plot(L[r], IL[r], label=DCBias[r])
        if i==2 :                                             # reference fitting
            plt.plot(L[-1], IL[-1], 'r', label='reference')
            plt.plot(L[-1], fit_IL, 'c-', label='{0}th Rsqure={1}'.format(6, Rref))
            plt.ylim(-60, -5)
        if i==3 :                                             # raw data - reference fitting
            for j in range(len(L) - 1):
                plt.plot(L[j], IL[j] - fit_IL, label=DCBias[j])
    plt.xlabel('Wavelength[nm]', fontsize=10)
    plt.ylabel('Transmission[dB]', fontsize=10)
    plt.title('Transmission spectra as measured', fontsize=12, fontweight='bold')
    plt.legend(loc='best', ncol=2, fontsize=8)
    plt.tight_layout()

    # save figure
    plt.suptitle(filename, fontsize=13, fontweight='bold')
    fig = plt.gcf()
    fig.set_size_inches((25, 17), forward=False)
    plt.savefig('{}/res/{}.png'.format(projectpath,filename))

    # csv
    TestSiteInfo = root.find('TestSiteInfo')
    AlignWavelength = next(root.iter('AlignWavelength'))
    Modulator = next(root.iter('Modulator'))
    Rsq = result1.rsquared                        # IV결정계수
    times = root.attrib['CreationDate']
    times = parse(times).strftime('%Y%m%d_%H%M%S')
    # 필요한 데이터 입력
    dict['Lot'].append(TestSiteInfo.attrib['Batch'])
    dict['Wafer'].append(TestSiteInfo.attrib['Wafer'])
    dict['Mask'].append(TestSiteInfo.attrib['Maskset'])
    dict['TestSite'].append(TestSiteInfo.attrib['TestSite'])
    dict['Name'].append(Modulator.attrib['Name'])
    dict['Date'].append(times)
    dict['Operator'].append(root.attrib['Operator'])
    dict['DieRow'].append(TestSiteInfo.attrib['DieRow'])
    dict['DieColumn'].append(TestSiteInfo.attrib['DieColumn'])
    dict['AnalysisWavelength (nm)'].append(AlignWavelength.text)
    dict['Rsq of Ref. spectrum (6th)'].append(Rref)
    dict['Max transmission of Ref. spec. (dB)'].append(max(IL[-1]))
    dict['Rsq of IV'].append(Rsq)
    dict['I at -1V[A]'].append(I[V.index(-1)])
    dict['I at 1V[A]'].append(I[V.index(1)])

frame = pd.DataFrame(dict)
frame.to_csv('Data.csv', index=False)