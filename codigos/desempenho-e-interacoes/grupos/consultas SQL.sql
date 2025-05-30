DROP FUNCTION IF EXISTS turno; 
DELIMITER $ 
CREATE FUNCTION turno(madrugada INTEGER, manha INTEGER, tarde INTEGER, noite INTEGER) RETURNS VARCHAR(45) 
READS SQL DATA 
BEGIN 
    IF (madrugada + manha + tarde + noite = 0) THEN 
        RETURN ''; 
    ELSEIF (manha >= madrugada and manha >= tarde and manha >= noite) THEN 
        RETURN 'MANHA'; 
    ELSEIF (tarde >= madrugada and tarde >= manha and tarde >= noite) THEN 
        RETURN 'TARDE'; 
    ELSEIF (noite >= madrugada and noite >= manha and noite >= tarde) THEN 
        RETURN 'NOITE'; 
    ELSE 
        RETURN 'MADRUGADA'; 
    END IF; 
END 
$ 
 
CREATE OR REPLACE VIEW professor AS 
    select distinct ce.user_id 
    from course_enrollments ce 
    inner join courses c on (c.id = ce.course_id) 
    where ce.role in ('teacher', 'tutor') and c.environment_id = 10 
    order by ce.user_id; 
    
CREATE OR REPLACE VIEW aluno AS 
    select distinct u.id, u.friends_count, c.id as course_id, c.name as course  
    from users u 
    inner join course_enrollments ce on (u.id = ce.user_id) 
    inner join courses c on (c.id = ce.course_id)
    where ce.role like 'member' and ce.course_id > 27 AND ce.course_id < 36 and u.id not in (select * from professor) 
    order by u.id; 
    
select concat_ws(", ", a.id, a.course_id, 
	(select count(f.id) 
    from friendships f 
    where f.user_id = a.id and f.status like 'accepted'), 
    
    (select count(f.id) 
    from friendships f 
    where f.user_id = a.id and f.status like 'accepted' and f.friend_id in (select id from aluno)), 
 
    (select concat_ws(", ", 
        IFNULL(CASE WHEN (m.recipient_id in (select id from aluno)) THEN count(distinct m.recipient_id) END, 0), 
        count(m.id), 
        IFNULL(sum(CASE WHEN (m.recipient_id in (select id from aluno)) THEN 1 END), 0), 
        IFNULL(sum(CASE WHEN (m.recipient_id in (select * from professor)) THEN 1 END), 0)) 
    from messages m where m.sender_id = a.id and m.created_at >= '2020-01-01'), 
    
    (select count(r.id) from results r where r.user_id = a.id and r.state like "finalized" and r.started_at >= '2020-01-01'), 
    
    (select concat_ws(", ", 
        IFNULL(sum(CASE WHEN st.type like "Help" THEN 1 END), 0), 
        IFNULL(sum(CASE WHEN st.type in ("Activity", "Answer") THEN 1 END), 0), 
        count(distinct str.id), 
        turno(IFNULL(sum(CASE WHEN (time(st.created_at) >= '00:00:00' and  time(st.created_at) <= '05:59:59') THEN 1 END), 0),  
            IFNULL(sum(CASE WHEN (time(st.created_at) >= '06:00:00' and  time(st.created_at) <= '11:59:59') THEN 1 END), 0), 
            IFNULL(sum(CASE WHEN (time(st.created_at) >= '12:00:00' and  time(st.created_at) <= '17:59:59') THEN 1 END), 0), 
            IFNULL(sum(CASE WHEN (time(st.created_at) >= '18:00:00' and  time(st.created_at) <= '23:59:59') THEN 1 END), 0)))  
    from statuses st 
    left join statuses str on (st.id = str.in_response_to_id and st.user_id <> str.user_id) 
    where st.created_at >= '2020-01-01' and st.user_id = a.id and st.type in ("Activity", "Answer", "Help")) 
) as "id_aluno, id_turma, var01, var02, var03, var04, var05, var06, var07, var08, var09, var10, var11" 
from aluno a;
