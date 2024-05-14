-- 1. creating database
create database project;



-- 2. using database 
use project;



-- 3. creating table
create table olympics
(
ID	int, 
Name varchar(400),
Sex	char(2),
Age int,
Height int,
Weight int,
Team varchar(400),
Noc	varchar(400),
Games varchar(400),
Year int,
Season varchar(400),
City varchar(400),
Sport varchar(400),
Event varchar(400),
Medal varchar(200)
);



-- 5. Inserting values into the table
load data infile 'D:\Athletes.csv'
into table olympics
fields terminated by ','
enclosed by '"'
lines terminated by '\r\n'
ignore 1 rows;



-- 6. CHECKING TABLE
select * from olympics;



-- 4. Showing how many medal counts present for entire data
select count(*) medal from olympics;



-- 5. Showing count of unique sports present in Olympics
select count(*) Sport from olympics;



-- 6. Showing how many different medals won by team India
select team, medal from olympics
where team = 'india';



-- 7. Show event wise medals won by India show from highest to lowest medals won in order
select team, event, medal from olympics
where team= 'india'
order by medal desc;



-- 8. showing country who won maximum medals.
select team,count(medal) as countm from olympics
group by team
order by countm desc
limit 1;



-- 9.showing top 10 countries who won gold
select team, medal as goldmedals from olympics
where medal='gold'
group by team
order by goldmedals desc
limit 10;



--  10. show in which year did United states won most gold medal
select team,year,count(medal) as most_gold_medal from olympics
where medal = 'gold' and team = 'united states'
group by team,year
order by most_gold_medal desc
limit 1;



-- 11. In which sports United States has most medals.
select team,sport, count(medal) as most_medals from olympics
where team='united states'
group by team,sport
order by most_medals desc
limit 1;



-- 12. Find top three players who won most medals along with their sports and country
select name, sport , team, count(medal) as most_medals_won from olympics
group by name,sport,team
order by most_medals_won desc
limit 3;



-- 13. Find player with most gold medals in cycling along with his country.
select name,team, sport,count(medal)as most_gold_medals from olympics
where medal='gold' and sport='cycling'
group by name,team,sport
order by most_gold_medals desc
limit 1;



-- 14.Find player with most medals (Gold+Silver+Bronze) in Basketball also show his country.
select name,sport, count(medal) as most_medals from olympics
where sport='basketball'
group by name,sport
order by most_medals desc
limit 1;



-- 15. Finding out the count of different medals of the top basketball player Teresa Edwards
select name,sport, count(medal) as count_of_medal from olympics
where name='teresa edwards'
group by name,sport
order by count_of_medal;



-- 16. Finding out medals won by male,female each year , Export this data and plot graph in excel
select sex,year,count(medal) as total_medal
from olympics
group by sex,year
order by total_medal;
