Author: Timur Abdualimov, leader SOVIET team

Competition: Recommended system, SkillFctory

First date code: 17.05.2020

Used: Kaggle notebook, GPU!


Цель - построить работающую рекомендательную систему.
Основной ноутбук: nn-colab-filter.pynb

В рамках модуля, в составе команды Soviet принимал участие в учебном соревновании на платформе Kaggle. 
"Glamour" with EDA on Kaggle: https://www.kaggle.com/abdualimov/nn-collab-filter
Ссылка на соревнование - https://www.kaggle.com/c/recommendationsv4/leaderboard.

В командной работе были опробованы основные библиотеки для пострения рекомендвательных систем - surprise, lightFM, LightGBM, stotlight, fast AI colab.

В реализации я остановил свой выбор в реализации коллаборативной фильтрации с использованием нейронной сети.
<img src= https://www.kaggleusercontent.com/kf/34926210/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..40QWV5vZeM2ad9kB9q6aaQ.3anHnFeSvuTIEodWcicp795WmzVQ0d_GbGRdRJYZK8utCFEtTWxHqcTfSbYiqMYV1Ti0iT4aYZZdYXlDdKL816FOIfFcwsoGqE7f2XrvKuUWzBh_oTlgoAu4KuvCWNVvl_-grefvcDhtDBjzqtOo_cz5jkccRIzkjPzr2smuedMNZh5SnSvcNVoVRrX2eabH_kHrkLYX4xFYZ1pHu6PW58lEat_YTCIxMv1TX93SEn-IvXuKZPjv0JapeOP0tMLtqbgdJKadWImtLIifEVJ_BKZnxk8r3hmDhZXmXKjr8QFvqOuG_62uwe6DmVw-ev91R0OZ5LnIGYWgJw4qdex_4sVCFu53btOBMvTGlMNfMeNsfwDJAc294PbLGgUZh5NqV2uqryNgMKTVfcaWw7_DZXdFOlMAzZgI2sQnudS3keeJnwNs7oBSAjIOlgFzdTJOTb3onLHUs4FJ4sFY0NdZUenrz1VIKiN0SCu1K6chjYaOeHaWAhPZQL87hjcDrk_6lk5JjzI9zFwoOd1_XcVyIeXjtYL6ZaSVa5GdS68SUSV-TBNy37ZFRk2QMImHPj0UIfugMi0lSmxoN9mHDUgPHikastB8HZJIF3NwMAEzwZg-Z_fmz2CzBq_VmS0tUW5xnktnD28jMoO2DTHHFQeq-al9vOicfzIktlH8lR1l6xA.hQzgfpsP1Ty-_W0TKp_o1w/__results___files/__results___12_0.png width="100%">

Также в целях ознакомления была представлен ноутбук с использованием fast ai.

Демонстрация работы с помощью streamlit and heroku: https://shielded-bastion-08476.herokuapp.com

* В папке Prod, содержится production код модели.
