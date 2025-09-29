import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
import pickle
import logging
from typing import List, Dict, Optional, Tuple
import requests
from io import BytesIO
from PIL import Image
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArcFaceService:
    def __init__(self):
        self.app = None
        self.face_database = {}
        self.embeddings_file = 'face_embeddings.pkl'
        self.similarity_threshold = 0.5
        self.supabase_service = None  # Will be injected
        
        self.initialize_model()
        
    def set_supabase_service(self, supabase_service):
        """Inject supabase service for database operations"""
        self.supabase_service = supabase_service
        
    def initialize_model(self):
        """Initialize the ArcFace model"""
        try:
            self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("ArcFace model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ArcFace model: {e}")
            raise e
    
    def load_face_database(self):
        """Load existing face database from file"""
        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, 'rb') as f:
                    self.face_database = pickle.load(f)
                logger.info(f"Loaded {len(self.face_database)} faces from database")
            except Exception as e:
                logger.error(f"Error loading face database: {e}")
                self.face_database = {}
        else:
            self.face_database = {}
    
    def save_face_database(self):
        """Save face database to file"""
        try:
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.face_database, f)
            logger.info(f"Saved {len(self.face_database)} faces to database")
        except Exception as e:
            logger.error(f"Error saving face database: {e}")
    
    def download_image(self, url: str) -> Optional[np.ndarray]:
        """Download image from URL and convert to OpenCV format"""
        try:
            # Check if this looks like a Supabase Storage path
            if not url.startswith('http') and self.supabase_service:
                # This might be a storage path, try to download from Supabase Storage
                logger.info(f"Attempting to download from Supabase Storage: {url}")
                
                # Try to download directly from storage
                file_data = self.supabase_service.download_storage_file(url, 'profile-images')
                if file_data:
                    image = Image.open(BytesIO(file_data))
                    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    logger.info(f"Successfully downloaded image from storage: {url}")
                    return opencv_image
                else:
                    # Fallback: try to get signed URL and download via HTTP
                    signed_url = self.supabase_service.get_storage_url(url, 'profile-images')
                    logger.info(f"Trying signed URL: {signed_url}")
                    url = signed_url
            
            # Standard HTTP download
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, timeout=30, headers=headers)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                # Convert PIL image to OpenCV format
                opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                logger.info(f"Successfully downloaded image from URL: {url[:100]}...")
                return opencv_image
            else:
                logger.error(f"Failed to download image from {url}, status code: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error downloading image from {url}: {e}")
            return None
    
    def extract_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding from image using ArcFace"""
        try:
            faces = self.app.get(image)
            if len(faces) > 0:
                # Get the face with highest confidence
                face = max(faces, key=lambda x: x.det_score)
                return face.normed_embedding
            else:
                logger.warning("No face detected in image")
                return None
        except Exception as e:
            logger.error(f"Error extracting face embedding: {e}")
            return None
    
    def extract_face_info(self, image: np.ndarray) -> Optional[Dict]:
        """Extract comprehensive face information including embedding and landmarks"""
        try:
            faces = self.app.get(image)
            if len(faces) > 0:
                # Get the face with highest confidence
                face = max(faces, key=lambda x: x.det_score)
                
                return {
                    'embedding': face.normed_embedding,
                    'bbox': face.bbox.tolist(),  # Bounding box [x1, y1, x2, y2]
                    'landmarks': face.kps.tolist() if hasattr(face, 'kps') and face.kps is not None else [],  # 5 facial landmarks
                    'confidence': float(face.det_score),
                    'age': int(face.age) if hasattr(face, 'age') else None,
                    'gender': face.sex if hasattr(face, 'sex') else None
                }
            else:
                logger.warning("No face detected in image")
                return None
        except Exception as e:
            logger.error(f"Error extracting face info: {e}")
            return None
    
    def register_user_face(self, user_id: str, user_data: Dict, image_url: str) -> bool:
        """Register a user's face in the database"""
        try:
            # Skip processing for placeholder/mock images
            if 'placeholder' in image_url.lower() or 'via.placeholder' in image_url.lower():
                logger.info(f"Skipping placeholder image for user {user_id}")
                # Create a mock embedding for development
                mock_embedding = np.random.rand(512).astype(np.float32)
                self.face_database[user_id] = {
                    'embedding': mock_embedding,
                    'user_data': user_data
                }
                logger.info(f"Registered mock face for user {user_id}")
                return True
            
            # Download and process image
            image = self.download_image(image_url)
            if image is None:
                return False
            
            # Extract face embedding
            embedding = self.extract_face_embedding(image)
            if embedding is None:
                return False
            
            # Store in database
            self.face_database[user_id] = {
                'embedding': embedding,
                'user_data': user_data
            }
            
            logger.info(f"Registered face for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering user face: {e}")
            return False
    
    def register_multiple_faces(self, users_data: List[Dict]) -> int:
        """Register multiple users' faces"""
        success_count = 0
        
        for user in users_data:
            if user.get('faceScannedUrl'):
                success = self.register_user_face(
                    user['id'], 
                    user, 
                    user['faceScannedUrl']
                )
                if success:
                    success_count += 1
        
        self.save_face_database()
        logger.info(f"Successfully registered {success_count} out of {len(users_data)} users")
        return success_count
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Normalize embeddings
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def recognize_face_from_image(self, image: np.ndarray) -> Optional[Dict]:
        try:
            print("\n" + "🔍 STARTING FACE RECOGNITION" + "\n")
            
            query_embedding = self.extract_face_embedding(image)
            if query_embedding is None:
                print("❌ No face detected in image\n")
                return None

            print(f"✅ Face detected, comparing with {len(self.face_database)} enrolled faces...")
            
            best_match = None
            best_similarity = 0.0

            for user_id, data in self.face_database.items():
                stored_embedding = data['embedding']
                similarity = self.calculate_similarity(query_embedding, stored_embedding)

                if similarity > best_similarity and similarity > self.similarity_threshold:
                    best_similarity = similarity
                    best_match = {
                        'user_id': user_id,
                        'user_data': data['user_data'],
                        'similarity': similarity
                    }

            if not best_match:
                print(f"❌ No match found (threshold: {self.similarity_threshold})\n")
                return None

            print("\n" + "="*80)
            print("🎯 FACE MATCH FOUND!")
            print(f"   User ID: {best_match['user_id']}")
            print(f"   Name: {best_match['user_data'].get('firstName')} {best_match['user_data'].get('lastName')}")
            print(f"   Email: {best_match['user_data'].get('email')}")
            print(f"   Similarity: {best_match['similarity']:.4f}")
            print("="*80 + "\n")

            if self.supabase_service:
                try:
                    actual_user_id = best_match['user_id']
                    
                    # Verify user exists
                    print(f"🔍 Verifying user {actual_user_id} exists in User table...")
                    user_check = self.supabase_service.supabase.table('User').select('id').eq('id', actual_user_id).execute()
                    print(f"   User check result: {user_check.data}\n")
                    
                    if not user_check.data:
                        print(f"❌ User {actual_user_id} NOT FOUND in User table!")
                        print("   Cannot save attendance - skipping\n")
                        return best_match
                    
                    print(f"✅ User exists in User table\n")
                    
                    # Log recognition
                    print("📝 Logging recognition event...")
                    recognition_data = {
                        'session_id': None,
                        'recognized_user_detail_id': best_match['user_id'],
                        'recognized_user_id': actual_user_id,
                        'similarity_score': best_match['similarity'],
                        'threshold_used': self.similarity_threshold,
                        'candidates_count': len(self.face_database),
                        'recognition_status': 'success',
                        'error_message': None,
                        'processing_time_ms': None,
                        'image_metadata': {},
                        'all_similarities': {},
                        'attendance_marked': False,
                        'attendance_id': None,
                        'ip_address': None,
                        'user_agent': None
                    }
                    
                    log_result = self.supabase_service.log_face_recognition(recognition_data)
                    print(f"   Log result: {log_result.get('success', False)}\n")

                    # Mark attendance
                    print("📋 Marking attendance...\n")
                    attendance_result = self.supabase_service.mark_attendance(
                        actual_user_id,
                        best_match['user_data']
                    )
                    
                    print("\n" + "="*80)
                    print("ATTENDANCE RESULT:")
                    print(f"   Success: {attendance_result.get('success')}")
                    print(f"   Message: {attendance_result.get('message')}")
                    print(f"   Existing: {attendance_result.get('existing', False)}")
                    print("="*80 + "\n")
                    
                except Exception as log_error:
                    print(f"\n❌ ERROR in recognition/attendance: {log_error}\n")
                    logger.exception(log_error)

            return best_match

        except Exception as e:
            print(f"\n❌ ERROR recognizing face: {e}\n")
            logger.exception(e)
            return None
    
    def recognize_face_from_base64(self, base64_image: str) -> Optional[Dict]:
        """Recognize face from base64 encoded image"""
        try:
            # Decode base64 image
            import base64
            image_data = base64.b64decode(base64_image.split(',')[1] if ',' in base64_image else base64_image)
            image = Image.open(BytesIO(image_data))
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Use real face recognition
            return self.recognize_face_from_image(opencv_image)
            
        except Exception as e:
            logger.error(f"Error recognizing face from base64: {e}")
            return None
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the face database"""
        return {
            'total_faces': len(self.face_database),
            'threshold': self.similarity_threshold
        }
    
    def enroll_face_from_base64(self, user_detail_id: str, user_id: str, base64_image: str, user_data: Dict = None) -> Dict:
        """Enroll a new face from base64 image with database integration"""
        start_time = datetime.now()
        
        try:
            # Decode base64 image
            import base64
            image_data = base64.b64decode(base64_image.split(',')[1] if ',' in base64_image else base64_image)
            image = Image.open(BytesIO(image_data))
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Extract face information
            face_info = self.extract_face_info(opencv_image)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if face_info is None:
                # Log failed enrollment
                if self.supabase_service:
                    self.supabase_service.log_face_enrollment(
                        user_detail_id, user_id,
                        {
                            'enrollment_status': 'failed',
                            'face_count_detected': 0,
                            'error_message': 'No face detected in image',
                            'image_source': 'api',
                            'processing_time_ms': int(processing_time)
                        }
                    )
                
                return {
                    'success': False,
                    'message': 'No face detected in image'
                }
            
            # Store in local database for quick access
            self.face_database[user_detail_id] = {
                'embedding': face_info['embedding'],
                'user_data': user_data or {},
                'face_info': {
                    'bbox': face_info['bbox'],
                    'landmarks': face_info['landmarks'],
                    'confidence': face_info['confidence'],
                    'age': face_info['age'],
                    'gender': face_info['gender']
                },
                'enrolled_at': datetime.now().isoformat(),
                'user_detail_id': user_detail_id,
                'user_id': user_id
            }
            
            # Save to file
            self.save_face_database()
            
            # Save to database if supabase service is available
            if self.supabase_service:
                # Save face embedding
                embedding_result = self.supabase_service.save_face_embedding(
                    user_detail_id, user_id,
                    {
                        'embedding': face_info['embedding'],
                        'confidence': face_info['confidence'],
                        'face_quality_score': face_info['confidence'],  # Use confidence as quality score
                        'source_url': user_data.get('faceScannedUrl', '') if user_data else '',
                        'enrollment_method': 'api'
                    }
                )
                
                # Save face landmarks
                landmarks_result = self.supabase_service.save_face_landmarks(
                    user_detail_id, user_id,
                    {
                        'landmarks': face_info['landmarks'],
                        'bbox': face_info['bbox'],
                        'age': face_info['age'],
                        'gender': face_info['gender']
                    }
                )
                
                # Log successful enrollment
                self.supabase_service.log_face_enrollment(
                    user_detail_id, user_id,
                    {
                        'enrollment_status': 'success',
                        'face_count_detected': 1,
                        'face_quality_score': face_info['confidence'],
                        'image_source': 'api',
                        'processing_time_ms': int(processing_time)
                    }
                )
            
            logger.info(f"Successfully enrolled face for user_detail_id {user_detail_id}")
            return {
                'success': True,
                'message': f'Face enrolled successfully for user {user_detail_id}',
                'face_info': {
                    'bbox': face_info['bbox'],
                    'landmarks': face_info['landmarks'],
                    'confidence': face_info['confidence'],
                    'age': face_info['age'],
                    'gender': face_info['gender']
                },
                'processing_time_ms': int(processing_time)
            }
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            error_msg = f"Error enrolling face: {str(e)}"
            logger.error(error_msg)
            
            # Log failed enrollment
            if self.supabase_service:
                self.supabase_service.log_face_enrollment(
                    user_detail_id, user_id,
                    {
                        'enrollment_status': 'failed',
                        'face_count_detected': 0,
                        'error_message': error_msg,
                        'image_source': 'api',
                        'processing_time_ms': int(processing_time)
                    }
                )
            
            return {
                'success': False,
                'message': error_msg
            }
    
    def remove_face(self, user_id: str) -> bool:
        """Remove a face from the database"""
        try:
            if user_id in self.face_database:
                del self.face_database[user_id]
                self.save_face_database()
                logger.info(f"Removed face for user {user_id}")
                return True
            else:
                logger.warning(f"User {user_id} not found in face database")
                return False
        except Exception as e:
            logger.error(f"Error removing face: {e}")
            return False
    
    def list_enrolled_faces(self) -> List[Dict]:
        """List all enrolled faces with their information"""
        try:
            enrolled_faces = []
            for user_id, data in self.face_database.items():
                face_entry = {
                    'user_id': user_id,
                    'enrolled_at': data.get('enrolled_at', 'Unknown'),
                    'user_data': data.get('user_data', {}),
                }
                
                # Add face info if available
                if 'face_info' in data:
                    face_entry['face_info'] = data['face_info']
                
                enrolled_faces.append(face_entry)
            
            return enrolled_faces
        except Exception as e:
            logger.error(f"Error listing enrolled faces: {e}")
            return []
    
    def update_threshold(self, new_threshold: float) -> bool:
        """Update similarity threshold"""
        try:
            if 0.0 <= new_threshold <= 1.0:
                self.similarity_threshold = new_threshold
                logger.info(f"Updated similarity threshold to {new_threshold}")
                return True
            else:
                logger.error("Threshold must be between 0.0 and 1.0")
                return False
        except Exception as e:
            logger.error(f"Error updating threshold: {e}")
            return False
    
    def sync_faces_from_database(self) -> Dict:
        """Sync and enroll faces from existing database images"""
        if not self.supabase_service:
            return {'success': False, 'message': 'Supabase service not available'}
        
        try:
            # Get users who need face enrollment
            users_to_enroll = self.supabase_service.get_users_for_face_enrollment()
            
            if not users_to_enroll:
                return {
                    'success': True,
                    'message': 'No users found requiring face enrollment',
                    'enrolled_count': 0,
                    'failed_count': 0
                }
            
            enrolled_count = 0
            failed_count = 0
            results = []
            
            logger.info(f"Starting face enrollment for {len(users_to_enroll)} users")
            
            for user in users_to_enroll:
                try:
                    user_detail_id = user['id']
                    user_id = user['userId']
                    face_url = user['faceScannedUrl']
                    
                    if not face_url or face_url.strip() == '':
                        failed_count += 1
                        results.append({
                            'user_detail_id': user_detail_id,
                            'status': 'failed',
                            'message': 'No face URL provided'
                        })
                        continue
                    
                    # Download and process the image
                    image = self.download_image(face_url)
                    if image is None:
                        failed_count += 1
                        results.append({
                            'user_detail_id': user_detail_id,
                            'status': 'failed',
                            'message': 'Failed to download image'
                        })
                        continue
                    
                    # Extract face information
                    face_info = self.extract_face_info(image)
                    if face_info is None:
                        failed_count += 1
                        results.append({
                            'user_detail_id': user_detail_id,
                            'status': 'failed',
                            'message': 'No face detected in image'
                        })
                        
                        # Log failed enrollment
                        self.supabase_service.log_face_enrollment(
                            user_detail_id, user_id,
                            {
                                'enrollment_status': 'failed',
                                'face_count_detected': 0,
                                'error_message': 'No face detected in image',
                                'image_source': 'sync_from_db',
                                'source_url': face_url
                            }
                        )
                        continue
                    
                    # Store in local database
                    self.face_database[user_detail_id] = {
                        'embedding': face_info['embedding'],
                        'user_data': user,
                        'face_info': {
                            'bbox': face_info['bbox'],
                            'landmarks': face_info['landmarks'],
                            'confidence': face_info['confidence'],
                            'age': face_info['age'],
                            'gender': face_info['gender']
                        },
                        'enrolled_at': datetime.now().isoformat(),
                        'user_detail_id': user_detail_id,
                        'user_id': user_id
                    }
                    
                    # Save to database
                    embedding_result = self.supabase_service.save_face_embedding(
                        user_detail_id, user_id,
                        {
                            'embedding': face_info['embedding'],
                            'confidence': face_info['confidence'],
                            'face_quality_score': face_info['confidence'],
                            'source_url': face_url,
                            'enrollment_method': 'sync_from_db'
                        }
                    )
                    
                    landmarks_result = self.supabase_service.save_face_landmarks(
                        user_detail_id, user_id,
                        {
                            'landmarks': face_info['landmarks'],
                            'bbox': face_info['bbox'],
                            'age': face_info['age'],
                            'gender': face_info['gender']
                        }
                    )
                    
                    # Log successful enrollment
                    self.supabase_service.log_face_enrollment(
                        user_detail_id, user_id,
                        {
                            'enrollment_status': 'success',
                            'face_count_detected': 1,
                            'face_quality_score': face_info['confidence'],
                            'image_source': 'sync_from_db',
                            'source_url': face_url
                        }
                    )
                    
                    enrolled_count += 1
                    results.append({
                        'user_detail_id': user_detail_id,
                        'status': 'success',
                        'confidence': face_info['confidence'],
                        'message': f"Successfully enrolled {user.get('firstName', '')} {user.get('lastName', '')}"
                    })
                    
                    logger.info(f"Successfully enrolled face for {user.get('firstName', '')} {user.get('lastName', '')} (ID: {user_detail_id})")
                    
                except Exception as e:
                    failed_count += 1
                    error_msg = f"Error processing user {user_detail_id}: {str(e)}"
                    logger.error(error_msg)
                    
                    results.append({
                        'user_detail_id': user_detail_id,
                        'status': 'failed',
                        'message': error_msg
                    })
                    
                    # Log failed enrollment
                    try:
                        self.supabase_service.log_face_enrollment(
                            user_detail_id, user.get('userId', ''),
                            {
                                'enrollment_status': 'failed',
                                'face_count_detected': 0,
                                'error_message': error_msg,
                                'image_source': 'sync_from_db',
                                'source_url': user.get('faceScannedUrl', '')
                            }
                        )
                    except:
                        pass  # Don't let logging errors stop the process
            
            # Save local database
            self.save_face_database()
            
            return {
                'success': True,
                'message': f'Face enrollment completed. Enrolled: {enrolled_count}, Failed: {failed_count}',
                'enrolled_count': enrolled_count,
                'failed_count': failed_count,
                'total_processed': len(users_to_enroll),
                'results': results
            }
            
        except Exception as e:
            error_msg = f"Error during face sync: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg
            }
    
    def load_embeddings_from_database(self) -> bool:
        """Load face embeddings from User -> UserDetails -> Storage flow"""
        if not self.supabase_service:
            return False
        
        try:
            # Get all VALID users (those in User table with UserDetails)
            users = self.supabase_service.get_all_users_with_profiles()
            self.face_database.clear()
            
            if not users:
                logger.warning("No valid users found in User -> UserDetails flow")
                return False
            
            # Process each valid user's face from storage
            for user in users:
                face_url = user.get('faceScannedUrl')  # Using faceScannedUrl from get_all_users_with_profiles
                if not face_url or not face_url.strip():
                    continue
                    
                user_id = user['id']
                logger.info(f"Loading face for valid user: {user.get('firstName', '')} {user.get('lastName', '')} - {user_id}")
                
                try:
                    # Download and process the face image from storage
                    image = self.download_image(face_url)
                    if image is None:
                        continue
                    
                    # Extract face embedding
                    embedding = self.extract_face_embedding(image)
                    if embedding is None:
                        continue
                    
                    # Store in face database with valid user data
                    self.face_database[user_id] = {
                        'embedding': embedding,
                        'user_data': user,
                        'confidence': 1.0
                    }
                    
                except Exception as e:
                    logger.error(f"Error processing face for user {user_id}: {e}")
                    continue
            
            logger.info(f"Loaded {len(self.face_database)} face embeddings from valid User -> UserDetails flow")
            return True
            
        except Exception as e:
            logger.error(f"Error loading embeddings from User -> UserDetails flow: {e}")
            return False
    
    def scan_and_enroll_from_database(self, supabase_service, save_to_db: bool = True) -> Dict:
        """Scan all face images from database and enroll them for recognition"""
        try:
            # Get all users with face URLs from database
            users = supabase_service.get_all_users_with_profiles()
            
            if not users:
                return {
                    'success': False,
                    'message': 'No users found in database'
                }
            
            results = {
                'total_users': len(users),
                'processed': 0,
                'enrolled': 0,
                'failed': 0,
                'errors': []
            }
            
            for user in users:
                face_url = user.get('faceScannedUrl')  # Use faceScannedUrl from get_all_users_with_profiles
                if not face_url or not face_url.strip():
                    continue
                
                results['processed'] += 1
                user_id = user['id']
                
                logger.info(f"Processing face for {user.get('firstName', '')} {user.get('lastName', '')} - {user_id}")
                
                try:
                    # Download and process the face image
                    image = self.download_image(face_url)
                    if image is None:
                        results['failed'] += 1
                        results['errors'].append(f"Could not download image for user {user_id}")
                        continue
                    
                    # Extract face information
                    face_info = self.extract_face_info(image)
                    if face_info is None:
                        results['failed'] += 1
                        results['errors'].append(f"No face detected for user {user_id}")
                        continue
                    
                    # Store in local face database (pickle file)
                    self.face_database[user_id] = {
                        'embedding': face_info['embedding'],
                        'user_data': user,
                        'face_info': {
                            'bbox': face_info['bbox'],
                            'landmarks': face_info['landmarks'],
                            'confidence': face_info['confidence'],
                            'age': face_info['age'],
                            'gender': face_info['gender']
                        },
                        'enrolled_at': datetime.now().isoformat(),
                        'source': 'database_scan'
                    }
                    
                    # Optionally save to database
                    if save_to_db:
                        embedding_data = {
                            'embedding': face_info['embedding'],
                            'user_detail_id': user.get('detail_id'),
                            'bbox': face_info['bbox'],
                            'confidence': face_info['confidence'],
                            'landmarks': face_info['landmarks'],
                            'age': face_info['age'],
                            'gender': face_info['gender'],
                            'source_image_url': face_url,
                            'enrollment_method': 'database_scan'
                        }
                        
                        supabase_service.save_face_embedding(user_id, embedding_data)
                    
                    results['enrolled'] += 1
                    logger.info(f"Successfully enrolled face for user {user_id}")
                    
                except Exception as user_error:
                    results['failed'] += 1
                    error_msg = f"Error processing user {user_id}: {str(user_error)}"
                    results['errors'].append(error_msg)
                    logger.error(error_msg)
            
            # Save the updated face database
            self.save_face_database()
            
            success_message = f"Database scan complete! Processed: {results['processed']}, Enrolled: {results['enrolled']}, Failed: {results['failed']}"
            logger.info(success_message)
            
            results.update({
                'success': True,
                'message': success_message
            })
            
            return results
            
        except Exception as e:
            error_msg = f"Error during database scan: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg
            }
