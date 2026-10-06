"""Finite JAX Verner 8(7)/9(8) controller and continuous output.

Ordinary one-output float64/complex128 sol structures only; events, mass,
NonNegative, output callbacks/refinement and single precision remain unported.
No fresh native MATLAB trajectory or paired-performance qualification.

Provenance
----------
MATLAB source: installed R2025b ode78.m/ode89.m and private/ntrp78.m/ntrp89.m.
Chebfun source context: @chebfun/ode78.m/@chebfun/ode89.m/@chebfun/odesol.m.
Chebfun commit: 7574c77
"""

import warnings
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp


def _stages78(fun, t, y, h, f1):
    f2 = fun(t + 0.05 * h, y + h * 0.05 * f1)
    f3 = fun(t + 0.1065625 * h, y + h * (-0.0069931640625 * f1 + 0.1135556640625 * f2))
    f4 = fun(t + 0.15984375 * h, y + h * (0.0399609375 * f1 + 0.1198828125 * f3))
    f5 = fun(
        t + 0.39 * h,
        y + h * (0.36139756280045754 * f1 + -1.3415240667004928 * f3 + 1.3701265039000352 * f4),
    )
    f6 = fun(
        t + 0.465 * h,
        y + h * (0.049047202797202795 * f1 + 0.23509720422144048 * f4 + 0.18085559298135673 * f5),
    )
    f7 = fun(
        t + 0.155 * h,
        y
        + h
        * (
            0.06169289044289044 * f1
            + 0.11236568314640277 * f4
            + -0.03885046071451367 * f5
            + 0.01979188712522046 * f6
        ),
    )
    f8 = fun(
        t + 0.943 * h,
        y
        + h
        * (
            -1.767630240222327 * f1
            + -62.5 * f4
            + -6.061889377376669 * f5
            + 5.6508231982227635 * f6
            + 65.62169641937624 * f7
        ),
    )
    f9 = fun(
        t + 0.901802041735857 * h,
        y
        + h
        * (
            -1.1809450665549708 * f1
            + -41.50473441114321 * f4
            + -4.434438319103725 * f5
            + 4.260408188586133 * f6
            + 43.75364022446172 * f7
            + 0.00787142548991231 * f8
        ),
    )
    f10 = fun(
        t + 0.909 * h,
        y
        + h
        * (
            -1.2814059994414884 * f1
            + -45.047139960139866 * f4
            + -4.731362069449577 * f5
            + 4.514967016593808 * f6
            + 47.44909557172985 * f7
            + 0.010592282971116612 * f8
            + -0.0057468422638446166 * f9
        ),
    )
    f11 = fun(
        t + 0.94 * h,
        y
        + h
        * (
            -1.7244701342624853 * f1
            + -60.92349008483054 * f4
            + -5.951518376222393 * f5
            + 5.556523730698456 * f6
            + 63.98301198033305 * f7
            + 0.014642028250414961 * f8
            + 0.06460408772358203 * f9
            + -0.0793032316900888 * f10
        ),
    )
    f12 = fun(
        t + h,
        y
        + h
        * (
            -3.301622667747079 * f1
            + -118.01127235975251 * f4
            + -10.141422388456112 * f5
            + 9.139311332232058 * f6
            + 123.37594282840426 * f7
            + 4.62324437887458 * f8
            + -3.3832777380682018 * f9
            + 4.527592100324618 * f10
            + -5.828495485811623 * f11
        ),
    )
    f13 = fun(
        t + h,
        y
        + h
        * (
            -3.039515033766309 * f1
            + -109.26086808941763 * f4
            + -9.290642497400293 * f5
            + 8.43050498176491 * f6
            + 114.20100103783314 * f7
            + -0.9637271342145479 * f8
            + -5.0348840888021895 * f9
            + 5.958130824002923 * f10
        ),
    )
    fC = (
        0.04427989419007951 * f1
        + 0.3541049391724449 * f6
        + 0.2479692154956438 * f7
        + -15.694202038838084 * f8
        + 25.084064965558564 * f9
        + -31.738367786260277 * f10
        + 22.938283273988784 * f11
        + -0.2361324633071542 * f12
    )
    fE = (
        3.272103901028776e-05 * f1
        + 0.0005046250618777735 * f6
        + -0.00012117235897844563 * f7
        + 20.142336771313868 * f8
        + -5.237178599439828 * f9
        + 8.156744408794658 * f10
        + -22.938283273988784 * f11
        + 0.2361324633071542 * f12
        + -0.36016794372897754 * f13
    )
    return (
        y + h * fC,
        fE,
        jnp.stack((f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13), axis=1),
    )


def _dense_stages78(fun, t, y, h, f):
    f1 = f[:, 0]
    f6 = f[:, 5]
    f7 = f[:, 6]
    f8 = f[:, 7]
    f9 = f[:, 8]
    f10 = f[:, 9]
    f11 = f[:, 10]
    f12 = f[:, 11]
    f14 = fun(
        t + h,
        y
        + h
        * (
            0.04427989419007951 * f1
            + 0.3541049391724449 * f6
            + 0.2479692154956438 * f7
            + -15.694202038838084 * f8
            + 25.084064965558564 * f9
            + -31.738367786260277 * f10
            + 22.938283273988784 * f11
            + -0.2361324633071542 * f12
        ),
    )
    f15 = fun(
        t + 0.3110177634953864 * h,
        y
        + h
        * (
            0.04620700646754963 * f1
            + 0.045039041608424805 * f6
            + 0.23368166977134244 * f7
            + 37.83901368421068 * f8
            + -15.949113289454246 * f9
            + 23.028368351816102 * f10
            + -44.85578507769412 * f11
            + -0.06379858768647444 * f12
            + -0.012595035543861663 * f14
        ),
    )
    f16 = fun(
        t + 0.1725 * h,
        y
        + h
        * (
            0.05037946855482041 * f1
            + 0.041098361310460796 * f6
            + 0.17180541533481958 * f7
            + 4.614105319981519 * f8
            + -1.7916678830853965 * f9
            + 2.531658930485041 * f10
            + -5.324977860205731 * f11
            + -0.03065532595385635 * f12
            + -0.005254479979429613 * f14
            + -0.08399194644224793 * f15
        ),
    )
    f17 = fun(
        t + 0.7846 * h,
        y
        + h
        * (
            0.0408289713299708 * f1
            + 0.4244479514247632 * f6
            + 0.23260915312752345 * f7
            + 2.677982520711806 * f8
            + 0.7420826657338945 * f9
            + 0.1460377847941461 * f10
            + -3.579344509890565 * f11
            + 0.11388443896001738 * f12
            + 0.012677906510331901 * f14
            + -0.07443436349946675 * f15
            + 0.047827480797578516 * f16
        ),
    )
    return jnp.stack((f1, f6, f7, f8, f9, f10, f11, f12, f14, f15, f16, f17), axis=1)


_BI78 = (
    (
        -7.238550783576432811855355839508646327161,
        11.15330887588935170976376962782446833855,
        2.34875229807309355640904629061136935335,
        -1027.321675339240679090464776362465090654,
        1568.546608927281956416687915664731868885,
        -2000.882061921041961546811133479107090218,
        1496.620400693446268810344884971434468267,
        -16.41320775560933621675902845723196069900,
        -4.29672443178246482824254064733546854251,
        -20.41628069294821485579834313809132051248,
        16.53007184264271512356106095760699278945,
        -18.63064171313429626683549958846959067803,
    ),
    (
        26.00913483254676138219215542805486438340,
        -91.7609656398961659890179437322816238711,
        -11.6724894172018429369093778842231443146,
        9198.71432360760879019681406218311101879,
        -13995.38852541600542155322174511897930298,
        17864.36380347691630038038755096765127729,
        -13397.55405171476021512904990709508924800,
        147.6097045407002371315249807692915435608,
        38.6444746111678092366406218271498656093,
        153.5213232524836445391962375168798263930,
        -96.6861433615782065041742809436987893361,
        164.1994112280183092456176460821337125030,
    ),
    (
        -50.23684777762566731759165474184543812128,
        291.7074241722059450113911477530513089255,
        -3.339139076505928386509206543237093540,
        -33189.78048157363822223641020734287802492,
        50256.2124698102445419491620666726469821,
        -64205.1907515562863000297926577113695108,
        48323.5602199437493999696912750109765015,
        -535.719963714732106447158760197417632645,
        -140.3503471762808981414524290552248895548,
        -436.5502610211220460266289847121377276100,
        268.959934219531723149495873437076657635,
        -579.272256249540441494196462569641132906,
    ),
    (
        52.12072084601022449485077581012685809554,
        -430.4096692910862817449451677633631387823,
        94.885262249720610030798242337479596095,
        57750.0831348887181073584126028277545727,
        -86974.5128036219909523950692144595063700,
        111224.8489930378077126420609392735999202,
        -84051.4283423393032636942266780744607468,
        938.286247077820650371318861625025573381,
        246.3954669697502467443139611011701827640,
        598.214644262650861959065070073603792110,
        -428.681909788964647271837835032326719249,
        980.198255708866731505258442280896479501,
    ),
    (
        -27.06472451211777193118825764262673140465,
        299.4531188198997479843407054776900024282,
        -143.071126583012024456409244370652716962,
        -47698.93315706261990169947144294597707756,
        71494.7977095997701213661747332399327008,
        -91509.3392102130338542605593697286718077,
        69399.8582111570893316100585838633124312,
        -779.438309639349328345148153897689081893,
        -205.8341686964167118696204191085878165880,
        -398.7823950071290897160364203878571043995,
        354.578231152433375494079868740183658991,
        -786.224179015513894176220583239056456901,
    ),
    (
        5.454547288952965694339504452480078562780,
        -79.78911199784015209705095616004766020335,
        61.0967097444217359754873031115590556707,
        14951.54365344033382142012769129774268946,
        -22324.57139433374168317029445568645401598,
        28594.46085938937782634638310955782423389,
        -21748.11815446623273761450332307272543593,
        245.4393970278627292916961100938952065362,
        65.44129872356201885836080588282812631205,
        104.0129692060648441002024406476025340187,
        -114.7001840640649599911246871588418008302,
        239.7294100413035911863764570341369884827,
    ),
)


def _stages89(fun, t, y, h, f1):
    f2 = fun(t + 0.04 * h, y + h * 0.04 * f1)
    f3 = fun(
        t + 0.09648736013787361 * h, y + h * (-0.01988527319182291 * f1 + 0.11637263332969652 * f2)
    )
    f4 = fun(
        t + 0.1447310402068104 * h, y + h * (0.0361827600517026 * f1 + 0.10854828015510781 * f3)
    )
    f5 = fun(
        t + 0.576 * h,
        y + h * (2.2721142642901775 * f1 + -8.526886447976398 * f3 + 6.830772183686221 * f4),
    )
    f6 = fun(
        t + 0.2272326564618766 * h,
        y + h * (0.050943855353893744 * f1 + 0.1755865049809071 * f4 + 0.0007022961270757468 * f5),
    )
    f7 = fun(
        t + 0.5407673435381234 * h,
        y
        + h
        * (
            0.1424783668683285 * f1
            + -0.35417994346686843 * f4
            + 0.07595315450295101 * f5
            + 0.6765157656337123 * f6
        ),
    )
    f8 = fun(
        t + 0.64 * h,
        y + h * (0.07111111111111111 * f1 + 0.32799092876058983 * f6 + 0.24089796012829906 * f7),
    )
    f9 = fun(
        t + 0.48 * h,
        y
        + h * (0.07125 * f1 + 0.32688424515752457 * f6 + 0.11561575484247544 * f7 + -0.03375 * f8),
    )
    f10 = fun(
        t + 0.06754 * h,
        y
        + h
        * (
            0.048226773224658105 * f1
            + 0.039485599804954 * f6
            + 0.10588511619346581 * f7
            + -0.021520063204743093 * f8
            + -0.10453742601833482 * f9
        ),
    )
    f11 = fun(
        t + 0.25 * h,
        y
        + h
        * (
            -0.026091134357549235 * f1
            + 0.03333333333333333 * f6
            + -0.1652504006638105 * f7
            + 0.03434664118368617 * f8
            + 0.1595758283215209 * f9
            + 0.21408573218281934 * f10
        ),
    )
    f12 = fun(
        t + 0.6770920153543243 * h,
        y
        + h
        * (
            -0.03628423396255659 * f1
            + -1.0961675974272087 * f6
            + 0.1826035504321331 * f7
            + 0.07082254444170684 * f8
            + -0.02313647018482431 * f9
            + 0.27112047263209327 * f10
            + 1.3081337494229808 * f11
        ),
    )
    f13 = fun(
        t + 0.8115 * h,
        y
        + h
        * (
            -0.5074635056416975 * f1
            + -6.631342198657237 * f6
            + -0.2527480100908801 * f7
            + -0.49526123800360955 * f8
            + 0.2932525545253887 * f9
            + 1.440108693768281 * f10
            + 6.237934498647056 * f11
            + 0.7270192054526987 * f12
        ),
    )
    f14 = fun(
        t + 0.906 * h,
        y
        + h
        * (
            0.6130118256955932 * f1
            + 9.088803891640463 * f6
            + -0.40737881562934486 * f7
            + 1.7907333894903747 * f8
            + 0.714927166761755 * f9
            + -1.438580857841723 * f10
            + -8.26332931206474 * f11
            + -1.5375705708088652 * f12
            + 0.34538328275648716 * f13
        ),
    )
    f15 = fun(
        t + h,
        y
        + h
        * (
            -1.2116979103438739 * f1
            + -19.055818715595954 * f6
            + 1.2630606753898752 * f7
            + -6.913916969178458 * f8
            + -0.676462266509498 * f9
            + 3.367860445026608 * f10
            + 18.00675164312591 * f11
            + 6.83882892679428 * f12
            + -1.0315164519219504 * f13
            + 0.41291062321306227 * f14
        ),
    )
    f16 = fun(
        t + h,
        y
        + h
        * (
            2.1573890074940536 * f1
            + 23.807122198095804 * f6
            + 0.8862779249216556 * f7
            + 13.139130397598764 * f8
            + -2.6044157092877147 * f9
            + -5.193859949783873 * f10
            + -20.412340711541507 * f11
            + -12.300856252505723 * f12
            + 1.5215530950085394 * f13
        ),
    )
    fC = (
        0.014588852784055396 * f1
        + 0.0020241978878893325 * f8
        + 0.21780470845697167 * f9
        + 0.12748953408543898 * f10
        + 0.2244617745463132 * f11
        + 0.1787254491259903 * f12
        + 0.07594344758096558 * f13
        + 0.12948458791975614 * f14
        + 0.029477447612619417 * f15
    )
    fE = (
        0.005757813768188949 * f1
        + 1.0675934530948108 * f8
        + -0.14099636134393978 * f9
        + -0.014411715396914925 * f10
        + 0.030796961251883033 * f11
        + -1.1613152578179067 * f12
        + 0.32221113486118586 * f13
        + -0.12948458791975614 * f14
        + -0.029477447612619417 * f15
        + 0.04932600711506839 * f16
    )
    return (
        y + h * fC,
        fE,
        jnp.stack((f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16), axis=1),
    )


def _dense_stages89(fun, t, y, h, f):
    f1 = f[:, 0]
    f8 = f[:, 7]
    f9 = f[:, 8]
    f10 = f[:, 9]
    f11 = f[:, 10]
    f12 = f[:, 11]
    f13 = f[:, 12]
    f14 = f[:, 13]
    f15 = f[:, 14]
    f17 = fun(
        t + h,
        y
        + h
        * (
            0.014588852784055396 * f1
            + 0.0020241978878893325 * f8
            + 0.21780470845697167 * f9
            + 0.12748953408543898 * f10
            + 0.2244617745463132 * f11
            + 0.1787254491259903 * f12
            + 0.07594344758096558 * f13
            + 0.12948458791975614 * f14
            + 0.029477447612619417 * f15
        ),
    )
    f18 = fun(
        t + 0.7421010083583088 * h,
        y
        + h
        * (
            0.015601405261088616 * f1
            + 0.26811643933275847 * f8
            + 0.1883053124587791 * f9
            + 0.12491991374610308 * f10
            + 0.2302302127814522 * f11
            + -0.13603122161327985 * f12
            + 0.07488659971306953 * f13
            + -0.02812840029795629 * f14
            + -0.023144557264819496 * f15
            + 0.027345304241113474 * f17
        ),
    )
    f19 = fun(
        t + 0.888 * h,
        y
        + h
        * (
            0.013111957218440684 * f1
            + -0.1464024265969827 * f8
            + 0.2471264389666796 * f9
            + 0.13113752030800324 * f10
            + 0.21705603469825827 * f11
            + 0.286753671376032 * f12
            + 0.02323311339149422 * f13
            + 0.05250677264199396 * f14
            + 0.0028339515860099506 * f15
            + -0.008502403851995712 * f17
            + 0.06914537026206649 * f18
        ),
    )
    f20 = fun(
        t + 0.696 * h,
        y
        + h
        * (
            0.013989212133617684 * f1
            + -0.031574065179505 * f8
            + 0.2271812513272158 * f9
            + 0.12894864109967866 * f10
            + 0.2216682589135277 * f11
            + 0.19483682365424806 * f12
            + 0.05740088404417653 * f13
            + 0.09008366542675955 * f14
            + 0.015791532088442122 * f15
            + -0.018991315059091858 * f17
            + -0.08830926811918835 * f18
            + -0.11502562032988092 * f19
        ),
    )
    f21 = fun(
        t + 0.487 * h,
        y
        + h
        * (
            0.016151472919007624 * f1
            + 0.08098685003242906 * f8
            + 0.12769162943069304 * f9
            + 0.12348143593834805 * f10
            + 0.233985125914011 * f11
            + -0.06595995683357368 * f12
            + -0.02565276859406433 * f13
            + -0.1258973463819247 * f14
            + -0.04307672490364844 * f15
            + 0.04973042479196705 * f17
            + 0.10004735401793927 * f18
            + 0.13786588067636232 * f19
            + -0.12235337700754625 * f20
        ),
    )
    return jnp.stack((f1, f8, f9, f10, f11, f12, f13, f14, f15, f17, f18, f19, f20, f21), axis=1)


_BI89 = (
    (
        -12.75304069282388950483064356409920964903,
        -0.7205785602508598770412906345635211707530,
        -48.06969107148755163304089843112677204750,
        16.32345788425353372538518290168630386345,
        -5.888504109270884968456670963647316074790,
        -69.22821100686856642029708151339949374410,
        -38.04668072585188932845088326881300565231,
        -75.21598899610186748511604166683735788867,
        -19.46588639117053206710108537195779499764,
        22.25276964616901764060657108171977843291,
        14.38227638804283974976194859491865842966,
        94.92756297288050252130347529152607755873,
        63.97757128518312942674385099931772857860,
        57.52494337729701822053356654527592436144,
    ),
    (
        68.54470113831162103032818060021729044674,
        6.559119452090996226782640921071801708575,
        451.8280048138745279924509263176669733181,
        -118.7000544943430560099961188668917723161,
        89.44704113715942232261735606938508401319,
        627.4402883568152894700875088172124339413,
        340.9894379782233715580222226199659613703,
        670.5551756563966247587443735545005782764,
        172.8442419404515527792492527636420780662,
        -202.2239493340537483960420170974810960961,
        -301.8862921284913174289877784920239768093,
        -819.3810521264963242304829331408227079441,
        -398.4710369466142415586467341825347335632,
        -587.5456254433247185141268798839079144120,
    ),
    (
        -194.8086610529652454702496599846926584629,
        -23.65417183348355244612060745867113623926,
        -1652.497181212881040972312045847339995538,
        379.7566858308294928207030358265089540700,
        -380.5212519133325379643523316755978956588,
        -2258.351433966898561277167615407280413114,
        -1221.089637320158271327850635113537223709,
        -2395.425253052829223230171636642290078449,
        -616.3040486605512589130986770456245791880,
        741.1380586926791133828821084525361362850,
        1781.158815560130838385479471257488378388,
        2811.287710727701073955763185548552769132,
        716.5966869905804904541294272473572154569,
        2312.713681211178682602365980842590527027,
    ),
    (
        317.7392440058917273091720330437842255568,
        44.16895609281556502164197054695369477469,
        3109.740640759030910750811309301753788556,
        -658.7110535057816543655898314260197627370,
        770.9937272473377188110779497353201847420,
        4212.395800215491820633428188331388768232,
        2271.121533555259071203608357605468537656,
        4449.143231627101738352915330060972953361,
        1143.483637444009748019174885933341490497,
        -1419.345991543901288934527221623894382397,
        -4706.785555404445608825447884178551282728,
        -4973.994583541372152306531194715881127387,
        52.89052166461839778508215175982407959882,
        -4612.840108616055993454816044374461167725,
    ),
    (
        -299.5110317016553611442365899609354277447,
        -45.39036357573745767095716350329312304563,
        -3211.322751886536222454319056864879758831,
        644.2805206048469902599556429935393520064,
        -831.4716893548379599332950159082545018560,
        -4325.916149192490627782429268979382924699,
        -2328.099120585713191256845645730711618239,
        -4556.769050179329341174034103718438911683,
        -1170.357669638643781845251579162353094100,
        1505.723453115174938364140876028053730124,
        6291.645831877559252216657958850293660213,
        4810.867980444981597632278117658725359609,
        -1470.500844962700814722191656818947551329,
        4986.820885035081979510527485116584809575,
    ),
    (
        151.9504248851541972968268022793780785647,
        24.43461389111572285100991773589739045465,
        1734.432866374712319698880133120955280438,
        -334.8751825767893992539894957157903467478,
        461.9387177871916189774255799856083869291,
        2327.648261456242912420669452255092833474,
        1251.132631830924456968695609885535799254,
        2447.365350086656490540478347199404246857,
        628.2905498131297391288540900112436704439,
        -841.3580375761078273196578619974820483141,
        -4164.432457468105482852522949304279052327,
        -2417.563170381452428143421579409061339218,
        1524.433133991196905505389714285761298379,
        -2793.397702113869225818637760332264198189,
    ),
    (
        -32.14704772912899411981910701782974118232,
        -5.395551268662524772664900940711910826177,
        -383.8940830682559717177907419834455933213,
        72.05311579106953179281845208383275272780,
        -104.2735790197010640588910140881002072916,
        -513.8098304131662767347811744202515986459,
        -275.9322212851025822454499429637433189491,
        -539.5239805539746656111293686438409662456,
        -138.4613470596128476846868080169758635556,
        193.8136970000397952625975451565478819654,
        1085.917381175309478755059233272153614833,
        493.8555519037577305710909287669609682492,
        -488.9260320222638668905067532907780371207,
        636.7239265496922574541536520861820193631,
    ),
)


class _RKState(NamedTuple):
    t: jax.Array
    y: jax.Array
    absh: jax.Array
    nfev: jax.Array
    nfailed: jax.Array
    done: jax.Array


class _RKTrial(NamedTuple):
    absh: jax.Array
    h: jax.Array
    done: jax.Array
    nfev: jax.Array
    nfailed: jax.Array
    nofailed: jax.Array
    accepted: jax.Array
    tolerance_failed: jax.Array
    invalid: jax.Array
    tnew: jax.Array
    ynew: jax.Array
    stages: jax.Array
    error: jax.Array


@partial(jax.jit, static_argnames=("fun", "solver_name", "norm_control"))
def _rk_step(
    fun,
    s,
    t0,
    f0,
    tfinal,
    direction,
    threshold,
    rtol,
    userhmin,
    userhmax,
    *,
    solver_name,
    norm_control,
):
    stages_fn = _stages78 if solver_name == "ode78" else _stages89
    dense_fn = _dense_stages78 if solver_name == "ode78" else _dense_stages89
    ns, extra, pow = (13, 4, 1 / 8) if solver_name == "ode78" else (16, 5, 1 / 9)
    hmin = jnp.maximum(16 * jnp.spacing(jnp.abs(s.t)), userhmin)
    hmax = jnp.maximum(16 * jnp.spacing(jnp.abs(s.t)), userhmax)
    absh = jnp.minimum(hmax, jnp.maximum(hmin, s.absh))
    done = 1.1 * absh >= jnp.abs(tfinal - s.t)
    h = jnp.where(done, tfinal - s.t, direction * absh)
    absh = jnp.abs(h)
    f1 = jax.lax.cond(s.t == t0, lambda _: f0, lambda _: fun(s.t, s.y), None)
    nfev = s.nfev + jnp.asarray(s.t != t0, jnp.int32)
    trial = _RKTrial(
        absh,
        h,
        done,
        nfev,
        s.nfailed,
        jnp.asarray(True),
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(False),
        s.t,
        s.y,
        jnp.zeros((s.y.size, ns), dtype=s.y.dtype),
        jnp.asarray(0.0),
    )

    def keep(a):
        return ~(a.accepted | a.tolerance_failed | a.invalid)

    def attempt(a):
        ynew, fE, stages = stages_fn(fun, s.t, s.y, a.h, f1)
        tnew = jnp.where(a.done, tfinal, s.t + a.h)
        h = tnew - s.t
        if norm_control:
            oldnorm = jnp.linalg.norm(s.y)
            scale = jnp.where(a.nofailed, jnp.maximum(oldnorm, jnp.linalg.norm(ynew)), oldnorm)
            err = a.absh * jnp.linalg.norm(fE) / jnp.maximum(scale, threshold)
        else:
            scale = jnp.where(a.nofailed, jnp.maximum(jnp.abs(s.y), jnp.abs(ynew)), jnp.abs(s.y))
            err = a.absh * jnp.max(jnp.abs(fE / jnp.maximum(scale, threshold)))
        invalid = ~(jnp.all(jnp.isfinite(ynew)) & jnp.all(jnp.isfinite(stages)) & jnp.isfinite(err))
        failed = ~(err <= rtol)
        tolerance_failed = failed & (a.absh <= hmin)
        reduced = jnp.where(
            a.nofailed,
            jnp.maximum(hmin, a.absh * jnp.maximum(0.1, 0.8 * (rtol / err) ** pow)),
            jnp.maximum(hmin, 0.5 * a.absh),
        )
        nextabsh = jnp.where(failed, reduced, a.absh)
        return _RKTrial(
            nextabsh,
            jnp.where(failed, direction * nextabsh, h),
            a.done & ~failed,
            a.nfev + jnp.asarray(ns - 1, jnp.int32),
            a.nfailed + jnp.asarray(failed, jnp.int32),
            a.nofailed & ~failed,
            ~failed & ~invalid,
            tolerance_failed,
            invalid,
            tnew,
            ynew,
            stages,
            err,
        )

    trial = jax.lax.while_loop(keep, attempt, trial)
    dense = jax.lax.cond(
        trial.accepted,
        lambda _: dense_fn(fun, s.t, s.y, trial.h, trial.stages),
        lambda _: jnp.zeros((s.y.size, 12 if solver_name == "ode78" else 14), dtype=s.y.dtype),
        None,
    )
    invalid = trial.invalid | ~jnp.all(jnp.isfinite(dense))
    growth = 1.25 * (trial.error / rtol) ** pow
    absh = jnp.where(
        trial.nofailed, jnp.where(growth > 0.2, trial.absh / growth, 5 * trial.absh), trial.absh
    )
    result = _RKState(
        trial.tnew,
        trial.ynew,
        absh,
        trial.nfev + jnp.asarray(trial.accepted, jnp.int32) * extra,
        trial.nfailed,
        trial.done,
    )
    return result, dense, trial.tolerance_failed, invalid


@partial(jax.jit, static_argnames=("solver_name",))
def _dense_point(ti, t, y, h, f, *, solver_name):
    cols = jnp.asarray(_BI78 if solver_name == "ode78" else _BI89, dtype=jnp.float64)
    theta = (ti - t) / h
    b = cols[-1]
    db = (cols.shape[0] + 1) * cols[-1]
    for i in range(cols.shape[0] - 2, -1, -1):
        b = b * theta + cols[i]
        db = db * theta + (i + 2) * cols[i]
    b = b * (theta * theta)
    b = b.at[0].add(theta)
    db = db * theta
    db = db.at[0].add(1.0)
    return y + h * (f @ b), f @ db


def native_rk(solver_name, odefun, tspan, y0, options=None, *, max_steps=100000):
    """Return a finite native Verner one-output structure on JAX CPU/x64.

    Provenance
    ----------
    MATLAB source: installed R2025b ode78.m/ode89.m, private/odearguments.m,
    private/ntrp78.m/private/ntrp89.m, private/odefinalize.m and deval.m.
    Chebfun integration context: @chebfun/ode78.m/@chebfun/ode89.m/@chebfun/odesol.m.
    Chebfun commit: 7574c77
    """
    if solver_name not in ("ode78", "ode89"):
        raise ValueError("native_rk solver_name must be ode78 or ode89")
    pow = 1 / 8 if solver_name == "ode78" else 1 / 9

    if not jax.config.x64_enabled:
        raise ValueError("native Verner RK port requires JAX x64")
    options = {} if options is None else dict(options)
    known = {"RelTol", "AbsTol", "InitialStep", "MaxStep", "MinStep", "NormControl"}

    def empty(v):
        return (
            v is None
            or (isinstance(v, (list, tuple, dict, str)) and not v)
            or getattr(v, "size", None) == 0
        )

    for key, value in options.items():
        if key not in known and not empty(value):
            raise NotImplementedError(f"native Verner RK option {key} is not yet implemented")

    def opt(key, default):
        value = options.get(key)
        return default if empty(value) else value

    # odearguments converts tspan with tspan(:), in column-major order.
    span = jnp.asarray(tspan, dtype=jnp.float64).reshape(-1, order="F")
    if span.ndim != 1 or span.size < 2 or not bool(jnp.all(jnp.isfinite(span))):
        raise ValueError("native Verner RK requires at least two finite times")
    direction = 1.0 if float(span[-1]) > float(span[0]) else -1.0
    if not bool(jnp.all(direction * jnp.diff(span) > 0)):
        raise ValueError("native Verner RK times must be strictly monotone")
    # Source y0(:) accepts row, column and matrix-shaped initial data.
    y = jnp.asarray(y0).reshape(-1, order="F")
    if y.ndim != 1 or y.size == 0 or not bool(jnp.all(jnp.isfinite(y))):
        raise ValueError("native Verner RK requires a finite initial state vector")
    y = y.astype(jnp.complex128 if jnp.iscomplexobj(y) else jnp.float64)

    def rhs(t, y):
        value = jnp.atleast_1d(jnp.asarray(odefun(t, y)))
        # Complex state with real derivative must keep both cached branches
        # in the promoted state dtype; infer complex RHS before this cast.
        return value.astype(jnp.result_type(y, value))

    f0 = rhs(span[0], y)
    if f0.shape != y.shape or not bool(jnp.all(jnp.isfinite(f0))):
        raise ValueError("native Verner RK RHS must return one finite derivative per component")
    y = y.astype(jnp.result_type(y, f0))
    f0 = f0.astype(y.dtype)
    rtol = jnp.asarray(opt("RelTol", 1e-3), dtype=jnp.float64)
    if rtol.ndim != 0 or not bool(jnp.isfinite(rtol) & (rtol > 0)):
        raise ValueError("RelTol must be a finite positive scalar")
    if float(rtol) < 100 * jnp.finfo(jnp.float64).eps:
        warnings.warn("RelTol increased to native100*eps floor", stacklevel=2)
        rtol = jnp.asarray(100 * jnp.finfo(jnp.float64).eps)
    atol = jnp.asarray(opt("AbsTol", 1e-6), dtype=jnp.float64).reshape(-1, order="F")
    norm_control = opt("NormControl", "off")
    if isinstance(norm_control, str):
        if norm_control.lower() not in ("on", "off"):
            raise ValueError("NormControl must be on or off")
        norm_control = norm_control.lower() == "on"
    norm_control = bool(norm_control)
    if atol.ndim != 1 or atol.size not in (1, y.size) or (norm_control and atol.size != 1):
        raise ValueError("AbsTol must be scalar or one per component; NormControl requires scalar")
    if not bool(jnp.all(jnp.isfinite(atol) & (atol > 0))):
        raise ValueError("AbsTol must be finite and positive")
    threshold = atol[0] / rtol if norm_control else jnp.broadcast_to(atol, y.shape) / rtol
    t0, tfinal = span[0], span[-1]
    tlen = jnp.abs(tfinal - t0)
    safehmax = 16 * jnp.finfo(jnp.float64).eps * jnp.maximum(jnp.abs(t0), jnp.abs(tfinal))
    userhmax = jnp.asarray(opt("MaxStep", jnp.maximum(0.1 * tlen, safehmax)), dtype=jnp.float64)
    userhmin = jnp.asarray(opt("MinStep", 0.0), dtype=jnp.float64)
    if not bool(jnp.isfinite(userhmax) & (userhmax > 0)) or not bool(
        jnp.isfinite(userhmin) & (userhmin >= 0)
    ):
        raise ValueError("MaxStep must be positive and MinStep nonnegative")
    if not empty(options.get("MinStep")) and float(userhmin) == 0:
        raise ValueError("explicit MinStep must be positive")
    userhmin = jnp.minimum(userhmin, tlen)
    if not empty(options.get("MaxStep")):
        userhmax = jnp.minimum(userhmax, tlen)
    if empty(options.get("MaxStep")):
        userhmax = jnp.maximum(userhmax, userhmin)
    elif float(userhmax) < float(userhmin):
        raise ValueError("MaxStep must be at least MinStep")
    tiny = 16 * jnp.spacing(jnp.abs(t0))
    hmin, hmax = jnp.maximum(tiny, userhmin), jnp.maximum(tiny, userhmax)
    initial = opt("InitialStep", None)
    if initial is None and float(userhmax) == float(userhmin):
        initial = userhmin
    if initial is None:
        absh = jnp.minimum(hmax, jnp.abs(span[1] - span[0]))
        if norm_control:
            rh = (jnp.linalg.norm(f0) / jnp.maximum(jnp.linalg.norm(y), threshold)) / (
                0.8 * rtol**pow
            )
        else:
            rh = jnp.max(jnp.abs(f0 / jnp.maximum(jnp.abs(y), threshold))) / (0.8 * rtol**pow)
        absh = jnp.where(absh * rh > 1, 1 / rh, absh)
        absh = jnp.maximum(absh, hmin)
    else:
        initial = jnp.abs(jnp.asarray(initial, dtype=jnp.float64))
        if initial.ndim != 0 or not bool(jnp.isfinite(initial) & (initial > 0)):
            raise ValueError("InitialStep must be finite and nonzero")
        absh = jnp.minimum(hmax, jnp.maximum(hmin, initial))

    state = _RKState(
        t0, y, absh, jnp.asarray(1, jnp.int32), jnp.asarray(0, jnp.int32), jnp.asarray(False)
    )
    history = [state]
    dense_history = [jnp.zeros((y.size, 12 if solver_name == "ode78" else 14), dtype=y.dtype)]
    for _ in range(max_steps):
        state, dense, tolerance_failed, invalid = _rk_step(
            rhs,
            state,
            t0,
            f0,
            tfinal,
            jnp.asarray(direction),
            threshold,
            rtol,
            userhmin,
            userhmax,
            solver_name=solver_name,
            norm_control=norm_control,
        )
        if bool(tolerance_failed):
            warnings.warn(
                "native Verner RK tolerance cannot be met at minimum step; "
                "returning accepted partial solution",
                stacklevel=2,
            )
            break
        if bool(invalid):
            raise RuntimeError("native Verner RK encountered nonfinite RHS/state")
        history.append(state)
        dense_history.append(dense)
        if bool(state.done):
            break
    else:
        raise RuntimeError("native Verner RK exceeded Python max_steps resource cap")
    mesh = jnp.stack([s.t for s in history])
    values = jnp.stack([s.y for s in history])
    stages = jnp.stack(dense_history)

    @partial(jax.jit, static_argnames=("return_derivative",))
    def dense_kernel(queries, *, return_derivative=False):
        if mesh.size == 1:
            if return_derivative:
                raise ValueError("native DEVAL derivative requires an accepted interval")
            return jnp.broadcast_to(values[0, :, None], (values.shape[1], queries.size))
        indices = jnp.clip(
            jnp.searchsorted(direction * mesh, direction * queries, side="right"), 1, mesh.size - 1
        )
        ys, dys = jax.vmap(
            lambda q, t, y, h, f: _dense_point(q, t, y, h, f, solver_name=solver_name)
        )(
            queries,
            mesh[indices - 1],
            values[indices - 1],
            mesh[indices] - mesh[indices - 1],
            stages[indices],
        )
        exact = jnp.clip(
            jnp.searchsorted(direction * mesh, direction * queries, side="left"), 0, mesh.size - 1
        )
        ys = jnp.where((queries == mesh[exact])[:, None], values[exact], ys)
        return (ys.T, dys.T) if return_derivative else ys.T

    def dense(times, *, return_derivative=False):
        queries = jnp.atleast_1d(jnp.asarray(times, dtype=jnp.float64))
        if queries.ndim != 1 or not bool(jnp.all(jnp.isfinite(queries))):
            raise ValueError("native DEVAL requires a finite time vector")
        if not bool(
            jnp.all(
                (direction * (queries - mesh[0]) >= 0) & (direction * (queries - mesh[-1]) <= 0)
            )
        ):
            raise ValueError("native DEVAL query outside integration interval")
        return dense_kernel(queries, return_derivative=return_derivative)

    return {
        "solver": solver_name,
        "x": mesh,
        "y": values.T,
        "sol": dense,
        "extdata": {"options": dict(options), "odefun": odefun, "varargin": ()},
        "ie": jnp.empty(0, dtype=jnp.int32),
        "xe": jnp.empty(0),
        "idata": {
            "f3d": jnp.moveaxis(stages, 0, -1),
            "idxNonNegative": jnp.empty(0, dtype=jnp.int32),
        },
        "stats": {
            "nsteps": len(history) - 1,
            "nfailed": int(state.nfailed),
            "nfevals": int(state.nfev),
            "tfinal": float(mesh[-1]),
        },
        "scope": "finite one-output native Verner; events/mass/nonnegative/output/singleprecision unported",
    }
