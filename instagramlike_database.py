import psycopg2
from psycopg2 import sql
import sys

print('program started')


# Connect to database
def connect_to_db():
    print('connect_to_db started')
    try:
        connection = psycopg2.connect(
            database="postgres",
            user="postgres",
            password="1111",
            host="127.0.0.1",
            port="5432"
        )
        print("connected", connection)
        return connection
    except Exception as error:
        print(f"Error: {error}")
        return None

'''
# Insertion (manually done and unused)
def add_user(username, password):
    connection = connect_to_db()
    if not connection:
        return
    try:
        hashed_pw = hash_password(password)
        cursor = connection.cursor()
        cursor.execute("INSERT INTO users (username, password) VALUES (postgres, postgres)", (username, hashed_pw))
        connection.commit()
        print(f"User {username} added successfully.")
    except Exception as e:
        print("Error adding user:", e)
    finally:
        cursor.close()
        connection.close()
'''

def authenticate_user(username, password):
    conn = connect_to_db()
    if not conn:
        return False, False, None
    try:
        cur = conn.cursor()
        # Check password using crypt
        cur.execute("""
            SELECT user_id, is_supervisor
            FROM users
            WHERE username = %s AND password = crypt(%s, password);
        """, (username, password))
        result = cur.fetchone()
        if result:
            if username == "Nikita" and password == '12345678':
                print("hi, The Nikita! you're allowed to request anything related to other users")
                return True, True, result[0]
            else:
                print(f"hi, The {username}! you're NOT allowed to request anything related to other users.")
                return True, False, result[0]
        else:
            return False, False, None
    except Exception as e:
        print("Authentication error:", e)
        return False, False, None
    finally:
        cur.close()
        conn.close()


# Non-supervisor allowed functions (also allowed for supervisor)
def view_own_posts(current_user_id):
    print('view_own_posts started')
    connection = connect_to_db()
    if not connection:
        return
    try:
        cursor = connection.cursor()
        cursor.execute("""
            SELECT post_id, caption, created_at
            FROM posts
            WHERE user_id = %s
            ORDER BY created_at DESC;
        """, (current_user_id,))
        posts = cursor.fetchall()
        for post in posts:
            print(f"Post ID: {post[0]}, Caption: {post[1]}, Created At: {post[2]}")
    except Exception as error:
        print("Error:", error)
    finally:
        cursor.close()
        connection.close()

def create_own_post(current_user_id, caption):
    print('create_own_post started')
    connection = connect_to_db()
    if not connection:
        return
    try:
        cursor = connection.cursor()
        cursor.execute(
        """
            INSERT INTO posts (user_id, caption) VALUES (%s, %s) RETURNING post_id;
        """, (current_user_id, caption))
        new_post_id = cursor.fetchone()[0]
        connection.commit()
        print(f"Post created with ID {new_post_id}")
    except Exception as error:
        print("Error:", error)
    finally:
        cursor.close()
        connection.close()

'''
def delete_own_post(current_user_id, post_id):
    print('delete_own_post started')
    connection = connect_to_db()
    if not connection:
        return
    try:
        cursor = connection.cursor()
        cursor.execute("""
            DELETE FROM posts
            WHERE post_id = %s AND user_id = %s
            RETURNING post_id;
        """, (post_id, current_user_id))
        deleted = cursor.fetchone()
        connection.commit()
        if deleted:
            print(f"Post {post_id} deleted successfully.")
        else:
            print("No post found or you don't have permission to delete this post.")
    except Exception as error:
        print("Error:", error)
    finally:
        cursor.close()
        connection.close()
'''

def delete_post(target_user_id, post_id, is_supervisor, current_user_id):
    if not is_supervisor and target_user_id != current_user_id:
        print("Access denied: Non-supervisor cannot delete another user's post.")
        return
    print(f"Attempting to delete post_id={post_id} of user_id={target_user_id}...")
    connection = connect_to_db()
    if not connection:
        return
    try:
        cursor = connection.cursor() # Verify the post belongs to the specified user_id
        cursor.execute("SELECT post_id FROM posts WHERE post_id = %s AND user_id = %s;", (post_id, target_user_id))
        post = cursor.fetchone()
        if not post:
            print("No such post found or it doesn't belong to the specified user.")
            cursor.close()
            connection.close()
            return

        cursor.execute("DELETE FROM posts WHERE post_id = %s;", (post_id,))
        connection.commit()
        print(f"Post {post_id} deleted successfully.")
    except Exception as error:
        print(f"Error deleting post: {error}")
    finally:
        cursor.close()
        connection.close()

'''
def delete_comment(target_user_id, comment_id, is_supervisor, current_user_id):
    if not is_supervisor and target_user_id != current_user_id:
        print("Access denied: Non-supervisor cannot delete another user's comment.")
        return

    print(f"Attempting to delete comment_id={comment_id} of user_id={target_user_id}...")
    connection = connect_to_db()
    if not connection:
        return
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT comment_id FROM comments WHERE comment_id = %s AND user_id = %s;", (comment_id, target_user_id))
        comment = cursor.fetchone()
        if not comment:
            print("No such comment found or it doesn't belong to the specified user.")
            cursor.close()
            connection.close()
            return
        cursor.execute("DELETE FROM comments WHERE comment_id = %s;", (comment_id,))
        connection.commit()
        print(f"Comment {comment_id} deleted successfully.")
    except Exception as error:
        print(f"Error deleting comment: {error}")
    finally:
        cursor.close()
        connection.close()
'''

def delete_account(username, is_supervisor, current_user_id):
    print(f"Attempting to delete account with username='{username}'...")
    connection = connect_to_db()
    if not connection:
        return
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT user_id, username FROM users WHERE username = %s;", (username,))
        user_row = cursor.fetchone()
        if not user_row:
            print("No user found with that username.")
            cursor.close()
            connection.close()
            return

        target_user_id = user_row[0]
        target_username = user_row[1]

        if not is_supervisor and target_user_id != current_user_id:
            print("Access denied: Non-supervisor cannot delete another user's account.")
            cursor.close()
            connection.close()
            return

        cursor.execute("DELETE FROM users WHERE user_id = %s;", (target_user_id,))
        connection.commit()
        print(f"Account '{target_username}' deleted successfully.")
    except Exception as error:
        print(f"Error deleting account: {error}")
    finally:
        cursor.close()
        connection.close()














# Supervisor allowed functions
# 1
def get_user_posts_with_engagement_rate(super_user_id, is_supervisor, current_user_id):
    if not is_supervisor and super_user_id != current_user_id:
        print("You do not have permission to view other users' posts.")
        return
    print(f'get_user_posts_with_engagement_rate started for user {current_user_id}')
    connection = connect_to_db()
    if not connection:
        return
    try:
        cursor = connection.cursor()
        query = sql.SQL(
            """
            SELECT p.post_id, p.caption, 
                   COUNT(DISTINCT l.like_id) AS like_count, 
                   COUNT(DISTINCT c.comment_id) AS comment_count
            FROM posts p
            LEFT JOIN likes l ON p.post_id = l.post_id
            LEFT JOIN comment c ON p.post_id = c.post_id
            WHERE p.user_id = %s
            GROUP BY p.post_id
            ORDER BY p.created_at DESC;
            """
        )
        print("Executing query to retrieve user's posts and engagement...")
        cursor.execute(query, (super_user_id,))
        posts = cursor.fetchall()

        if not posts:
            print("No data found for this user.")
        else:
            print(f"Found {len(posts)} posts for user_id={super_user_id}:")
            for post in posts:
                print(f"Post ID: {post[0]}, Caption: {post[1]}, Likes: {post[2]}, Comments: {post[3]}")

    except Exception as error:
        print(f"Error retrieving user posts with engagement: {error}")
    finally:
     cursor.close()
     connection.close()

# 2
def get_mutual_followers(super_user_id, is_supervisor, current_user_id):
    if not is_supervisor and super_user_id != current_user_id:
        print("You do not have permission to view other users' mutual followers.")
        return
    print(f'get_mutual_followers started user_id={current_user_id}')

    connection = connect_to_db()
    if not connection:
        return
    try:
        cursor = connection.cursor()
        query = sql.SQL(
            """
            SELECT u.username
            FROM follows f1
            JOIN follows f2 ON f1.follower_id = f2.followed_id AND f1.followed_id = f2.follower_id
            JOIN users u ON f1.follower_id = u.user_id
            WHERE f2.followed_id = %s;
            """
        )
        print("Executing query for mutual followers...")
        cursor.execute(query, (current_user_id,))
        mutual_followers = cursor.fetchall()

        if not mutual_followers:
            print("No mutual followers found.")
        else:
            print(f"Found {len(mutual_followers)} mutual followers:")
            for follower in mutual_followers:
                print(f"Mutual Follower: {follower[0]}")
    except Exception as error:
        print(f"Error retrieving mutual followers: {error}")
    finally:
        cursor.close()
        connection.close()


# 3
def get_posts_by_followed_users_in_location(super_user_id, location_name, is_supervisor, current_user_id):
    if not is_supervisor and super_user_id != current_user_id:
        print("You do not have permission to view other users' mutual followers.")
        return
    print(f'get_posts_by_followed_users_in_location started for user_id={current_user_id} in location {location_name}')

    connection = connect_to_db()
    if not connection:
        return

    try:
        cursor = connection.cursor()
        query = sql.SQL(
            """
            SELECT p.post_id, p.caption, p.created_at, loc.location_name
            FROM posts p
            JOIN follows f ON p.user_id = f.followed_id
            JOIN locations loc ON p.location_id = loc.location_id
            WHERE f.follower_id = %s AND loc.location_name = %s
            ORDER BY p.created_at DESC;
            """
        )
        print("Executing query for posts by followed users in location...")
        cursor.execute(query, (current_user_id, location_name))
        posts = cursor.fetchall()

        if not posts:
            print("No posts found for this location or the user does not follow anyone posting here.")
        else:
            print(f"Found {len(posts)} posts:")
            for post in posts:
                print(f"Post ID: {post[0]}, Caption: {post[1]}, Created At: {post[2]}, Location: {post[3]}")
    except Exception as error:
        print(f"Error retrieving posts by followed users in location: {error}")
    finally:
        cursor.close()
        connection.close()

# 4
def get_top_trending_hashtags(super_user_id, is_supervisor, current_user_id):
    if not is_supervisor and super_user_id != current_user_id:
        print("You do not have permission to view other users' mutual followers.")
        return
    print('get_top_trending_hashtags started')

    connection = connect_to_db()
    if not connection:
        return

    try:
        cursor = connection.cursor()
        query = sql.SQL(
            """
            SELECT h.tag, COUNT(ph.hashtag_id) AS usage_count
            FROM post_hashtags ph
            JOIN hashtags h ON ph.hashtag_id = h.hashtag_id
            JOIN posts p ON ph.post_id = p.post_id
            WHERE p.created_at >= NOW() - INTERVAL '7 days'
            GROUP BY h.tag
            ORDER BY usage_count DESC
            LIMIT 10;
            """
        )
        cursor.execute(query, is_supervisor)
        hashtags = cursor.fetchall()

        for hashtag in hashtags:
            print(f"Hashtag: {hashtag[0]}, Usage Count: {hashtag[1]}")
    except Exception as error:
        print(f"Error: {error}")
    finally:
        cursor.close()
        connection.close()

# 5
def get_most_active_users(super_user_id, is_supervisor, current_user_id):
    if not is_supervisor and super_user_id != current_user_id:
        print("You do not have permission to view other users' mutual followers.")
        return
    print('get_most_active_users started')
    connection = connect_to_db()
    if not connection:
        return

    try:
        cursor = connection.cursor()
        query = sql.SQL(
            """
            SELECT u.username, COUNT(DISTINCT p.post_id) AS post_count, COUNT(DISTINCT c.comment_id) AS comment_count
            FROM users u
            LEFT JOIN posts p ON u.user_id = p.user_id
            LEFT JOIN comment c ON u.user_id = c.user_id
            GROUP BY u.username
            ORDER BY (COUNT(DISTINCT p.post_id) + COUNT(DISTINCT c.comment_id)) DESC
            LIMIT 10;
            """
        )
        print("Executing query for most active users...")
        cursor.execute(query)
        active_users = cursor.fetchall()

        if not active_users:
            print("No active users found.")
        else:
            print(f"Found {len(active_users)} active users:")
            for user in active_users:
                print(f"Username: {user[0]}, Posts: {user[1]}, Comments: {user[2]}")

    except Exception as error:
     print(f"Error retrieving most active users: {error}")
    finally:
        cursor.close()
        connection.close()
# 6
def get_saved_posts_by_user(super_user_id, is_supervisor, current_user_id):
    if not is_supervisor and super_user_id != current_user_id:
        print("You do not have permission to view other users' mutual followers.")
        return
    print(f"you're in get_saved_posts_by_user user_id={current_user_id}")

    connection = connect_to_db()
    if not connection:
        return

    try:
        cursor = connection.cursor()
        query = sql.SQL(
            """
            SELECT p.post_id, p.caption, u.username AS saved_by
            FROM saved_posts sp
            JOIN posts p ON sp.post_id = p.post_id
            JOIN users u ON sp.user_id = u.user_id
            WHERE sp.user_id = %s;
            """
        )
        cursor.execute(query, (super_user_id,))
        saved_posts = cursor.fetchall()
        if not saved_posts:
            print("No saved posts found for this user.")
        else:
            print(f"Found {len(saved_posts)} saved posts:")
            for post in saved_posts:
                print(f"Post ID: {post[0]}, Caption: {post[1]}, Saved By: {post[2]}")
    except Exception as error:
        print(f"Error retrieving saved posts by user: {error}")
    finally:
        cursor.close()
        connection.close()

# 7
def find_top_5_posts(super_user_id, is_supervisor, current_user_id):
    if not is_supervisor and super_user_id != current_user_id:
        print("You do not have permission to view other users' mutual followers.")
        return
    print(f"you're in find_top_5_posts user_id={current_user_id}")

    connection = connect_to_db()
    if not connection:
        return

    try:
        cursor = connection.cursor()
        query = sql.SQL(
            """
            SELECT p.post_id, p.caption,
                   COUNT(DISTINCT l.like_id) AS like_count,
                   COUNT(DISTINCT c.comment_id) AS comment_count,
                   ((COUNT(DISTINCT l.like_id) + COUNT(DISTINCT c.comment_id))::FLOAT
                    / GREATEST(COUNT(DISTINCT f.follower_id), 1)) AS engagement_rate
            FROM posts p
            LEFT JOIN likes l ON p.post_id = l.post_id
            LEFT JOIN comment c ON p.post_id = c.post_id
            LEFT JOIN follows f ON f.followed_id = p.user_id
            WHERE p.user_id = %s
            GROUP BY p.post_id, p.caption
            ORDER BY engagement_rate DESC
            LIMIT 5;
            """
        )
        print("Executing query for top 5 posts by engagement...")
        cursor.execute(query, (current_user_id,))
        top_5_posts = cursor.fetchall()

        if not top_5_posts:
            print("No posts found for this user.")
        else:
            print(f"Found {len(top_5_posts)} top posts:")
            for post in top_5_posts:
                print(f"Post ID: {post[0]}, Caption: {post[1]}, Likes: {post[2]}, Comments: {post[3]}, Engagement Rate: {post[4]}")
    except Exception as error:
        print(f"Error retrieving top 5 posts: {error}")
    finally:
        cursor.close()
        connection.close()

# 8
def the_same_hashtags_for_posts(post_id, super_user_id, is_supervisor, current_user_id):
    if not is_supervisor and super_user_id != current_user_id:
        print("You do not have permission to view other users' mutual followers.")
        return
    print(f"you're in the_same_hashtags_for_posts post_id={post_id}")

    connection = connect_to_db()
    if not connection:
        return

    try:
        cursor = connection.cursor()
        query = sql.SQL(
            """
            SELECT DISTINCT p2.post_id, p2.caption
            FROM post_hashtags ph1
            JOIN post_hashtags ph2 ON ph1.hashtag_id = ph2.hashtag_id
            JOIN posts p2 ON ph2.post_id = p2.post_id
            WHERE ph1.post_id = %s
            AND p2.post_id != ph1.post_id;
            """
        )
        print("Executing query for posts with shared hashtags...")
        cursor.execute(query, (post_id,))
        same_hashtag_posts = cursor.fetchall()

        if not same_hashtag_posts:
            print("No other posts share hashtags with this post.")
        else:
            print(f"Found {len(same_hashtag_posts)} posts with shared hashtags:")
            for post in same_hashtag_posts:
                print(f"Post ID: {post[0]}, Caption: {post[1]}")
    except Exception as error:
        print(f"Error retrieving posts with same hashtags: {error}")
    finally:
        cursor.close()
        connection.close()


'''
# 8
def the_same_hashtags_for_posts(post_id):
    connection = connect_to_db()
    if not connection:
        return

    try:
        cursor = connection.cursor()
        query = sql.SQL(
            """
            SELECT DISTINCT p2.post_id, p2.caption
            FROM post_hashtags ph1
            JOIN post_hashtags ph2 ON ph1.hashtag_id = ph2.hashtag_id
            JOIN posts p2 ON ph2.post_id = p2.post_id
            WHERE ph1.post_id = %s
            AND p2.post_id != ph1.post_id;
            """
        )
        cursor.execute(query, (post_id,))
        same_hashtag_posts = cursor.fetchall()
        for post in same_hashtag_posts:
            print(f"Post ID: {post[0]}, Caption: {post[1]}")
    except Exception as error:
        print(f"Error: {error}")
    finally:
        cursor.close()
        connection.close()
'''


# 9
def who_comment_more(super_user_id, is_supervisor, current_user_id):
    if not is_supervisor and super_user_id != current_user_id:
        print("You do not have permission to view other users' mutual followers.")
        return
    print("you're in who_comment_more")
    connection = connect_to_db()
    if not connection:
        return

    try:
        cursor = connection.cursor()
        query = sql.SQL(
            """
            SELECT c1.user_id AS commenter_1, c2.user_id AS commenter_2, COUNT(DISTINCT c1.post_id) AS common_posts
            FROM comment c1
            JOIN comment c2 ON c1.post_id = c2.post_id AND c1.user_id != c2.user_id
            GROUP BY c1.user_id, c2.user_id
            HAVING COUNT(DISTINCT c1.post_id) > 3
            ORDER BY common_posts DESC;
            """
        )
        print("Executing query to find who comments more in common posts...")
        cursor.execute(query)
        pairs = cursor.fetchall()

        if not pairs:
            print("No pairs of users found who comment together on more than 3 posts.")
        else:
            print(f"Found {len(pairs)} pairs of users:")
            for pair in pairs:
                print(f"Commenter 1: {pair[0]}, Commenter 2: {pair[1]}, Common Posts: {pair[2]}")
    except Exception as error:
        print(f"Error retrieving comment pairs: {error}")
    finally:
        cursor.close()
        connection.close()


# 10
def never_post_but_active(super_user_id, is_supervisor, current_user_id):
    if not is_supervisor and super_user_id != current_user_id:
        print("You do not have permission to view other users' mutual followers.")
        return
    print("you're in never_post_but_active")
    connection = connect_to_db()
    if not connection:
        return

    try:
        cursor = connection.cursor()
        query = sql.SQL(
            """
            SELECT u.user_id, u.username,
                   COUNT(DISTINCT l.like_id) AS total_likes,
                   COUNT(DISTINCT c.comment_id) AS total_comment
            FROM users u
            LEFT JOIN posts p ON u.user_id = p.user_id
            LEFT JOIN likes l ON u.user_id = l.user_id
            LEFT JOIN comment c ON u.user_id = c.user_id
            WHERE p.post_id IS NULL
            GROUP BY u.user_id, u.username
            HAVING (COUNT(DISTINCT l.like_id) > 0 OR COUNT(DISTINCT c.comment_id) > 0);
            """
        )
        print("Executing query for never-post-but-active users...")
        cursor.execute(query)
        users_active_no_posts = cursor.fetchall()

        if not users_active_no_posts:
            print("No users found who never posted but are still active.")
        else:
            print(f"Found {len(users_active_no_posts)} such users:")
            for usr in users_active_no_posts:
                print(f"User ID: {usr[0]}, Username: {usr[1]}, Total Likes: {usr[2]}, Total Comments: {usr[3]}")
    except Exception as error:
        print(f"Error retrieving never-post-but-active users: {error}")
    finally:
        cursor.close()
        connection.close()


# 11
def who_like_or_comment_own_posts(super_user_id, is_supervisor, current_user_id):
    if not is_supervisor and super_user_id != current_user_id:
        print("You do not have permission to view other users' mutual followers.")
        return
    print("you're in who_like_or_comment_own_posts")
    connection = connect_to_db()
    if not connection:
        return

    try:
        cursor = connection.cursor()
        query = sql.SQL(
            """
            SELECT u.username, p.post_id, p.caption,
                   COUNT(DISTINCT l.like_id) AS self_likes,
                   COUNT(DISTINCT c.comment_id) AS self_comment
            FROM users u
            JOIN posts p ON u.user_id = p.user_id
            LEFT JOIN likes l ON p.post_id = l.post_id AND l.user_id = u.user_id
            LEFT JOIN comment c ON p.post_id = c.post_id AND c.user_id = u.user_id
            GROUP BY u.username, p.post_id, p.caption
            HAVING (COUNT(DISTINCT l.like_id) > 0 OR COUNT(DISTINCT c.comment_id) > 0);
            """
        )
        print("Executing query for self-liking or self-commenting users...")
        cursor.execute(query)
        results = cursor.fetchall()

        if not results:
            print("No users found who like or comment on their own posts.")
        else:
            print(f"Found {len(results)} users/posts combos:")
            for r in results:
                print(f"Username: {r[0]}, Post ID: {r[1]}, Caption: {r[2]}, Self Likes: {r[3]}, Self Comments: {r[4]}")
    except Exception as error:
        print(f"Error retrieving self-liking/commenting users: {error}")
    finally:
        cursor.close()
        connection.close()

# 12
def most_popular_location(super_user_id, is_supervisor, current_user_id):
    if not is_supervisor and super_user_id != current_user_id:
        print("You do not have permission to view other users' mutual followers.")
        return
    print("you're in most_popular_location")
    connection = connect_to_db()
    if not connection:
        return

    try:
        cursor = connection.cursor()
        query = sql.SQL(
            """
            SELECT loc.location_name, COUNT(p.post_id) AS post_count
            FROM locations loc
            JOIN posts p ON loc.location_id = p.location_id
            GROUP BY loc.location_name
            ORDER BY post_count DESC
            LIMIT 10;
            """
        )
        print("Executing query for most popular locations...")
        cursor.execute(query)
        locations = cursor.fetchall()

        if not locations:
            print("No locations found or no posts associated with locations.")
        else:
            print(f"Found {len(locations)} popular locations:")
            for loc in locations:
                print(f"Location: {loc[0]}, Post Count: {loc[1]}")
    except Exception as error:
        print(f"Error retrieving most popular locations: {error}")
    finally:
        cursor.close()
        connection.close()



# 13
def expired_stories_without_views(super_user_id, is_supervisor, current_user_id):
    if not is_supervisor and super_user_id != current_user_id:
        print("You do not have permission to view other users' mutual followers.")
        return
    print("you're in expired_stories_without_views")
    connection = connect_to_db()
    if not connection:
        return

    try:
        cursor = connection.cursor()
        query = sql.SQL(
            """
            SELECT s.story_id, s.image_url, s.video_url, s.created_at, s.expires_at
            FROM stories s
            LEFT JOIN story_views sv ON s.story_id = sv.story_id
            WHERE sv.view_id IS NULL
              AND s.expires_at < NOW();
            """
        )
        print("Executing query for expired stories without views...")
        cursor.execute(query)
        stories = cursor.fetchall()

        if not stories:
            print("No expired stories found without views.")
        else:
            print(f"Found {len(stories)} expired stories without views:")
            for st in stories:
                print(
                    f"Story ID: {st[0]}, Image URL: {st[1]}, Video URL: {st[2]}, Created At: {st[3]}, Expires At: {st[4]}")
    except Exception as error:
        print(f"Error retrieving expired stories: {error}")
    finally:
        cursor.close()
        connection.close()
# 14
def top_users_who_tag_others(super_user_id, is_supervisor, current_user_id):
    if not is_supervisor and super_user_id != current_user_id:
        print("You do not have permission to view other users' mutual followers.")
        return
    print("you  are in top_users_who_tag_others ")

    connection = connect_to_db()
    if not connection:
        return

    try:
        cursor = connection.cursor()
        query = sql.SQL(
            """
            SELECT u.username, COUNT(DISTINCT tu.tagged_user_id) AS total_tags
            FROM users u
            JOIN posts p ON u.user_id = p.user_id
            JOIN tagged_users tu ON p.post_id = tu.post_id
            GROUP BY u.username
            ORDER BY total_tags DESC
            LIMIT 10;
            """
        )
        cursor.execute(query)
        users_tags = cursor.fetchall()
        for ut in users_tags:
            print(f"Username: {ut[0]}, Total Tags: {ut[1]}")
    except Exception as error:
        print(f"Error: {error}")
    finally:
        cursor.close()
        connection.close()

# 15
def longest_comment_chain(super_user_id, is_supervisor, current_user_id):
    if not is_supervisor and super_user_id != current_user_id:
        print("You do not have permission to view other users' mutual followers.")
        return
    print("you  are in longest_comment_chain ")
    connection = connect_to_db()
    if not connection:
        return

    try:
        cursor = connection.cursor()
        query = sql.SQL(
            """
            WITH RECURSIVE comment_thread AS (
                SELECT comment_id, parent_comment_id, 1 AS depth
                FROM comment
                WHERE parent_comment_id IS NULL
                UNION ALL
                SELECT c.comment_id, c.parent_comment_id, ct.depth+1
                FROM comment c
                JOIN comment_thread ct ON c.parent_comment_id = ct.comment_id
            )
            SELECT MAX(depth) AS longest_chain
            FROM comment_thread;
            """
        )
        cursor.execute(query)
        result = cursor.fetchone()
        print(f"Longest Comment Chain: {result[0]}")
    except Exception as error:
        print(f"Error: {error}")
    finally:
        cursor.close()
        connection.close()

# 16
def find_posts_saved_by_the_same_users(super_user_id, is_supervisor, current_user_id):
    if not is_supervisor and super_user_id != current_user_id:
        print("You do not have permission to view other users' mutual followers.")
        return
    print("you're in find_posts_saved_by_the_same_users")

    connection = connect_to_db()
    if not connection:
        return

    try:
        cursor = connection.cursor()
        query = sql.SQL(
            """
            SELECT sp1.post_id AS post_1, sp2.post_id AS post_2, COUNT(DISTINCT sp1.user_id) AS common_saves
            FROM saved_posts sp1
            JOIN saved_posts sp2 ON sp1.user_id = sp2.user_id AND sp1.post_id < sp2.post_id
            GROUP BY sp1.post_id, sp2.post_id
            HAVING COUNT(DISTINCT sp1.user_id) > 2
            ORDER BY common_saves DESC;
            """
        )
        cursor.execute(query)
        pairs = cursor.fetchall()
        for pair in pairs:
            print(f"Post 1: {pair[0]}, Post 2: {pair[1]}, Common Saves: {pair[2]}")
    except Exception as error:
        print(f"Error: {error}")
    finally:
        cursor.close()
        connection.close()

# how to make requests secure
def get_user_posts_secure(super_user_id, is_supervisor, current_user_id):
    if not is_supervisor and super_user_id != current_user_id:
        print("You do not have permission to view other users' mutual followers.")
        return
    print("you're in get_user_posts_secure")
    connection = connect_to_db()
    if not connection:
        return

    try:
        cursor = connection.cursor()
        query = """
            SELECT p.post_id, p.caption
            FROM posts p
            WHERE p.user_id = %s
        """
        cursor.execute(query, (super_user_id,))
        posts = cursor.fetchall()
        for post in posts:
            print(f"Post ID: {post[0]}, Caption: {post[1]}")
    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    super_user_id = 7 # particular case for my insertion
    print("input your login(username):")
    username_input = input().strip()
    print("input your password:")
    password_input = input().strip()

    authenticated, is_supervisor, current_user_id = authenticate_user(username_input, password_input)
    print('authenticated', authenticated)
    print('is_supervisor', is_supervisor)
    if not authenticated:
        print("Authentication failed. Exiting.")
        sys.exit(1)

    if not is_supervisor:
        print("you are  allowed  the write down he following commands:\n"
              "# 1 view_own_posts\n"
              "# 2 create_own_post\n"
              "# 3 delete_own_post\n"
              "# delete_post 11, 101, is_supervisor, current_user_id\n"
              "# delete_account 'john_doe', is_supervisor, current_user_id\n"
              )

    if is_supervisor:
        print("you are  allowed  the write down he following commands:\n"
              "# 1 get_user_posts_with_engagement_rate 11\n"
              "# 2 get_mutual_followers 11\n"
              "# 3 get_posts_by_followed_users_in_location 11 NewYork\n"
              "# 4 get_top_trending_hashtags\n"
              "# 5 get_most_active_users\n"
              "# 6  get_saved_posts_by_user 11\n"
              "# 7 find_top_5_posts 11\n"
              "# 8 the_same_hashtags_for_posts 5\n"
              "# 9 who_comment_more\n"
              "# 10 never_post_but_active\n"
              "# 11 who_like_or_comment_own_posts\n"
              "# 12 most_popular_location\n"
              "# 13 expired_stories_without_views\n"
              "# 14 top_users_who_tag_others\n"
              "# 15 longest_comment_chain\n"
              "# 16 find_posts_saved_by_the_same_users\n"
              "# 16+  get_user_posts_secure 11\n\n\n"
              "# delete_post 11, 101, is_supervisor, current_user_id\n"
              "# delete_account 'john_doe', is_supervisor, current_user_id\n"
              )

    if len(sys.argv) < 2:
        print("Enter command:")
        command_line = input().strip()
        args = command_line.split()
        if len(args) == 0:
            print("No command provided.")
            sys.exit(1)
        sys.argv = [sys.argv[0]] + args

    command = sys.argv[1]

    conn = connect_to_db()
    cur = conn.cursor()
    cur.execute("SELECT user_id FROM users WHERE username = %s", (username_input,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        print("Error: could not find current user in DataBase.")
        sys.exit(1)
    current_user_id = row[0]

    # Adding user
    #if command == "add_user":
    #    if len(sys.argv) != 4:
    #        print("Usage: python script.py add_user <username> <password>")
    #    else:
    #        add_user(sys.argv[2], sys.argv[3])



    # 1
    if command == "get_user_posts_with_engagement_rate":
        if len(sys.argv) != 3:
            print("Usage: python script.py get_user_posts_with_engagement_rate <user_id>")
        else:
            get_user_posts_with_engagement_rate(int(sys.argv[2]), is_supervisor, current_user_id) # get_user_posts_with_engagement_rate 11
            print("get_user_posts_with_engagement_rate finished correctly")

    # 2
    if command == "get_mutual_followers":
        if len(sys.argv) != 3:
            print("Usage: python script.py get_mutual_followers <user_id>")
        else:
            get_mutual_followers(int(sys.argv[2]), is_supervisor, current_user_id) # get_mutual_followers 11
            print("get_mutual_followers finished correctly")


    # 3
    if command == "get_posts_by_followed_users_in_location":
        if len(sys.argv) != 4:
            print("Usage: python script.py get_posts_by_followed_users_in_location <user_id> <location_name>")
        else:
            get_posts_by_followed_users_in_location(int(sys.argv[2]), sys.argv[3], is_supervisor, current_user_id) # get_posts_by_followed_users_in_location 11 NewYork
            print("get_posts_by_followed_users_in_location finished correctly")


    # 4
    if command == "get_top_trending_hashtags":
        get_top_trending_hashtags(7, is_supervisor, current_user_id) # get_top_trending_hashtags
        print("get_top_trending_hashtags finished correctly")

    # 5
    if command == "get_most_active_users":
        get_most_active_users(7, is_supervisor, current_user_id) # get_most_active_users
        print("get_most_active_users finished correctly")


    # 6
    if command == "get_saved_posts_by_user":
        if len(sys.argv) != 3:
            print("Usage: python script.py get_saved_posts_by_user <user_id>")
        else:
            get_saved_posts_by_user(int(sys.argv[2]), is_supervisor, current_user_id) # get_saved_posts_by_user 11
            print("get_saved_posts_by_user finished correctly")


    # 7
    if command == "find_top_5_posts":
        if len(sys.argv) != 3:
            print("Usage: python script.py find_top_5_posts <user_id>")
        else:
            find_top_5_posts(int(sys.argv[2]), is_supervisor, current_user_id) # find_top_5_posts 11
            print("find_top_5_posts finished correctly")


    # 8
    if command == "the_same_hashtags_for_posts":
        if len(sys.argv) != 3:
            print("Usage: python script.py the_same_hashtags_for_posts <post_id>")
        else:
            the_same_hashtags_for_posts(int(sys.argv[2]), 7, is_supervisor, current_user_id) #  the_same_hashtags_for_posts 5
            print("the_same_hashtags_for_posts finished correctly")


    # 9
    if command == "who_comment_more":
        who_comment_more(7, is_supervisor, current_user_id) # who_comment_more
        print("who_comment_more finished correctly")


    # 10
    if command == "never_post_but_active":
        never_post_but_active(7, is_supervisor, current_user_id) # never_post_but_active
        print("never_post_but_active finished correctly")


    # 11
    if command == "who_like_or_comment_own_posts":
        who_like_or_comment_own_posts(7, is_supervisor, current_user_id) # who_like_or_comment_own_posts
        print("who_like_or_comment_own_posts finished correctly")


    # 12
    if command == "most_popular_location":
        most_popular_location(7, is_supervisor, current_user_id) # most_popular_location
        print("most_popular_location finished correctly")


    # 13
    if command == "expired_stories_without_views":
        expired_stories_without_views(7, is_supervisor, current_user_id) # expired_stories_without_views
        print("expired_stories_without_views finished correctly")


    # 14
    if command == "top_users_who_tag_others":
        top_users_who_tag_others(7, is_supervisor, current_user_id) # top_users_who_tag_others
        print("top_users_who_tag_others finished correctly")


    # 15
    if command == "longest_comment_chain":
        longest_comment_chain(7, is_supervisor, current_user_id) # longest_comment_chain
        print("longest_comment_chain finished correctly")


    # 16
    if command == "find_posts_saved_by_the_same_users":
        find_posts_saved_by_the_same_users(7, is_supervisor, current_user_id) # find_posts_saved_by_the_same_users
        print("find_posts_saved_by_the_same_users finished correctly")


    if command == "get_user_posts_secure":
        if len(sys.argv) != 3:
            print("Usage: python script.py get_user_posts_secure <user_id>")
        else:
            get_user_posts_secure(int(sys.argv[2]), is_supervisor, current_user_id) # get_user_posts_secure 11
            print("get_user_posts_secure finished correctly")

    # Non-supervisor user
    if command == "view_own_posts":
        if not is_supervisor:
            view_own_posts(current_user_id)
        else:
            view_own_posts(current_user_id)

    if command == "create_own_post":
        if len(sys.argv) < 3:
            print("Usage: create_own_post <caption>")
        else:
            if is_supervisor:
                caption = " ".join(sys.argv[2:])
                create_own_post(current_user_id, caption)
            else:
                caption = " ".join(sys.argv[2:])
                create_own_post(current_user_id, caption)


    if command == "delete_post": # delete_post 12 24-26
        if len(sys.argv) != 4:
            print("Usage: delete_post <user_id> <post_id>")
        else:
            delete_post(int(sys.argv[2]), int(sys.argv[3]), is_supervisor, current_user_id)

    '''
    elif command == "delete_comment":
        if len(sys.argv) != 4:
            print("Usage: delete_comment <user_id> <comment_id>")
        else:
            delete_comment(int(sys.argv[2]), int(sys.argv[3]), is_supervisor, current_user_id)
'''

    if command == "delete_account": # delete_account not_john_doe or travel_guru or cat_lover
        if len(sys.argv) != 3:
            print("Usage: delete_account <username>")
        else:
            delete_account(sys.argv[2], is_supervisor, current_user_id)




    else:
        if is_supervisor:
            print("Unknown command, but you are supervisor and can implement more.")
        else:
            print("Unknown or restricted command for non-supervisor user.")

